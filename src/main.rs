use std::io::Cursor;
use std::sync::mpsc;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound;
use reqwest::multipart;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up CPAL audio input.
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");
    let config = device.default_input_config()?;
    
    let sample_rate = config.sample_rate().0 as usize;
    let channels = config.channels() as usize;
    
    println!("Input device: {}", device.name()?);
    println!(
        "Using sample rate: {} Hz, channels: {}, sample format: {:?}",
        sample_rate,
        channels,
        config.sample_format()
    );
    
    // Create an MPSC channel to pass audio samples (as i16) from the capture callback.
    let (tx, rx) = mpsc::channel::<Vec<i16>>();
    
    // Error callback.
    let err_fn = |err| {
        eprintln!("Audio stream error: {}", err);
    };
    
    // Build the input stream based on the device's sample format.
    let stream = match config.sample_format() {
        cpal::SampleFormat::I16 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[i16], _| {
                    // Directly pass the i16 samples.
                    let samples = data.to_vec();
                    let _ = tx.send(samples);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::F32 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    // Convert each f32 sample to i16.
                    let samples: Vec<i16> =
                        data.iter().map(|&s| cpal::Sample::from_sample(s)).collect();
                    let _ = tx.send(samples);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[u16], _| {
                    // Convert each u16 sample to i16.
                    let samples: Vec<i16> =
                        data.iter().map(|&s| cpal::Sample::from_sample(s)).collect();
                    let _ = tx.send(samples);
                },
                err_fn,
                None,
            )?
        }
        _ => {
            eprintln!("Unsupported sample format");
            return Ok(());
        }
    };
    
    stream.play()?;
    println!("Audio stream started. Recording and processing in chunks...");
    
    // Determine how many samples to capture per chunk.
    // For example, capture 3 seconds of audio.
    let chunk_duration_secs = 3;
    let samples_per_chunk = sample_rate * chunk_duration_secs * channels;
    
    // Buffer to accumulate samples.
    let mut audio_buffer: Vec<i16> = Vec::with_capacity(samples_per_chunk);
    
    // Retrieve the API key from the environment.
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("Please set the OPENAI_API_KEY environment variable");
    
    // Main processing loop.
    loop {
        // Block briefly waiting for new audio samples.
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                audio_buffer.extend(samples);
                // When we have enough samples, process the chunk.
                if audio_buffer.len() >= samples_per_chunk {
                    // Extract one chunk.
                    let chunk: Vec<i16> = audio_buffer.drain(..samples_per_chunk).collect();
                    println!("Processing a chunk ({} samples)...", chunk.len());
                    
                    // Create an in‑memory WAV file from the chunk.
                    let wav_data = create_wav_in_memory(&chunk, channels as u16, sample_rate as u32)?;
                    
                    // Clone API key for the async task.
                    let api_key_clone = api_key.clone();
                    // Spawn an async task to send the chunk to the Whisper API.
                    tokio::spawn(async move {
                        match send_audio_chunk(wav_data, &api_key_clone).await {
                            Ok(transcription) => {
                                println!("Transcription: {}", transcription);
                            }
                            Err(e) => {
                                eprintln!("Error sending audio chunk: {}", e);
                            }
                        }
                    });
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    
    Ok(())
}

/// Create an in‑memory WAV file (as Vec<u8>) from a slice of i16 samples.
fn create_wav_in_memory(
    samples: &[i16],
    channels: u16,
    sample_rate: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Use a Cursor over a Vec<u8> as the output.
    let mut buffer = Cursor::new(Vec::<u8>::new());
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(&mut buffer, spec)?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    Ok(buffer.into_inner())
}

/// Send the given WAV audio data to the Whisper API and return the transcription text.
async fn send_audio_chunk(
    audio_data: Vec<u8>,
    api_key: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    // Build a multipart form with the audio file and required parameters.
    let form = multipart::Form::new()
        .part(
            "file",
            multipart::Part::bytes(audio_data)
                .file_name("chunk.wav")
                .mime_str("audio/wav")?,
        )
        .text("model", "whisper-1");
    
    let response = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .await?;
    
    // Parse the JSON response.
    let json: serde_json::Value = response.json().await?;
    // The transcription text should be in the "text" field.
    let transcription = json["text"].as_str().unwrap_or("").to_string();
    
    Ok(transcription)
}
