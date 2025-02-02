use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::sync::mpsc;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound;
use reqwest::multipart;

/// Compute the RMS (root-mean-square) energy of a block of i16 samples.
fn compute_rms(samples: &[i16]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Write a transcription to the FIFO so goose can read it.
fn write_to_fifo(transcription: &str) {
    // Attempt to open the FIFO for writing. This will block if no reader is attached.
    if let Ok(mut fifo) = OpenOptions::new().write(true).open("/tmp/goose_pipe") {
        if let Err(e) = writeln!(fifo, "{}", transcription) {
            eprintln!("Error writing to FIFO: {}", e);
        }
    } else {
        eprintln!("Could not open FIFO /tmp/goose_pipe for writing");
    }
}

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

    // Create an MPSC channel to receive audio sample blocks (converted to i16).
    let (audio_tx, rx) = mpsc::channel::<Vec<i16>>();
    let err_fn = |err| {
        eprintln!("Audio stream error: {}", err);
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::I16 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[i16], _| {
                    let samples = data.to_vec();
                    let _ = audio_tx.send(samples);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::F32 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    let samples: Vec<i16> =
                        data.iter().map(|&s| cpal::Sample::from_sample(s)).collect();
                    let _ = audio_tx.send(samples);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            device.build_input_stream(
                &config.into(),
                move |data: &[u16], _| {
                    let samples: Vec<i16> =
                        data.iter().map(|&s| cpal::Sample::from_sample(s)).collect();
                    let _ = audio_tx.send(samples);
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
    println!("Audio stream started. Using adaptive chunking...");

    // Adaptive chunking parameters.
    let energy_threshold = 300.0; // Adjust based on your microphone sensitivity.
    let min_speech_duration_secs = 0.5; // Minimum duration for a valid segment.
    let silence_duration_secs = 0.3;    // Duration of sustained silence to mark the end.
    let max_chunk_duration_secs = 10.0; // Maximum duration before forcing a flush.
    let min_speech_samples = (sample_rate as f64 * channels as f64 * min_speech_duration_secs) as usize;
    let silence_duration_samples = (sample_rate as f64 * channels as f64 * silence_duration_secs) as usize;
    let max_chunk_samples = (sample_rate as f64 * channels as f64 * max_chunk_duration_secs) as usize;
    // For a 100 ms period, the number of samples.
    let silence_increment = (sample_rate as f64 * channels as f64 * 0.1) as usize;

    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("Please set the OPENAI_API_KEY environment variable");

    // Adaptive chunking state.
    let mut in_speech = false;
    let mut current_chunk: Vec<i16> = Vec::new();
    let mut silence_samples = 0;

    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(block) => {
                let rms = compute_rms(&block);
                if !in_speech {
                    if rms >= energy_threshold {
                        in_speech = true;
                        silence_samples = 0;
                        current_chunk.extend(block);
                    }
                } else {
                    current_chunk.extend(&block);
                    if rms < energy_threshold {
                        silence_samples += block.len();
                    } else {
                        silence_samples = 0;
                    }
                    // End the segment if enough silence is detected.
                    if silence_samples >= silence_duration_samples {
                        if current_chunk.len() >= min_speech_samples {
                            println!("Finalizing chunk ({} samples)", current_chunk.len());
                            let wav_data = match create_wav_in_memory(
                                &current_chunk,
                                channels as u16,
                                sample_rate as u32,
                            ) {
                                Ok(data) => data,
                                Err(e) => {
                                    eprintln!("Error creating WAV in memory: {}", e);
                                    current_chunk.clear();
                                    in_speech = false;
                                    silence_samples = 0;
                                    continue;
                                }
                            };
                            let api_key_clone = api_key.clone();
                            tokio::spawn(async move {
                                match send_audio_chunk(wav_data, &api_key_clone).await {
                                    Ok(transcription) => {
                                        println!("Transcription: {}", transcription);
                                        if let Some(filtered) = filter_transcription(&transcription) {
                                            write_to_fifo(&filtered);
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("Error sending audio chunk: {}", e);
                                    }
                                }
                            });
                        }
                        current_chunk.clear();
                        in_speech = false;
                        silence_samples = 0;
                    }
                    // Flush if the chunk becomes too long.
                    if current_chunk.len() >= max_chunk_samples {
                        println!("Finalizing long chunk ({} samples)", current_chunk.len());
                        let wav_data = match create_wav_in_memory(
                            &current_chunk,
                            channels as u16,
                            sample_rate as u32,
                        ) {
                            Ok(data) => data,
                            Err(e) => {
                                eprintln!("Error creating WAV in memory: {}", e);
                                current_chunk.clear();
                                in_speech = false;
                                silence_samples = 0;
                                continue;
                            }
                        };
                        let api_key_clone = api_key.clone();
                        tokio::spawn(async move {
                            match send_audio_chunk(wav_data, &api_key_clone).await {
                                Ok(transcription) => {
                                    println!("Transcription: {}", transcription);
                                    if let Some(filtered) = filter_transcription(&transcription) {
                                        write_to_fifo(&filtered);
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Error sending audio chunk: {}", e);
                                }
                            }
                        });
                        current_chunk.clear();
                        in_speech = false;
                        silence_samples = 0;
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if in_speech {
                    silence_samples += silence_increment;
                    if silence_samples >= silence_duration_samples {
                        if current_chunk.len() >= min_speech_samples {
                            println!("Finalizing chunk after timeout ({} samples)", current_chunk.len());
                            let wav_data = match create_wav_in_memory(
                                &current_chunk,
                                channels as u16,
                                sample_rate as u32,
                            ) {
                                Ok(data) => data,
                                Err(e) => {
                                    eprintln!("Error creating WAV in memory: {}", e);
                                    current_chunk.clear();
                                    in_speech = false;
                                    silence_samples = 0;
                                    continue;
                                }
                            };
                            let api_key_clone = api_key.clone();
                            tokio::spawn(async move {
                                match send_audio_chunk(wav_data, &api_key_clone).await {
                                    Ok(transcription) => {
                                        println!("Transcription: {}", transcription);
                                        if let Some(filtered) = filter_transcription(&transcription) {
                                            write_to_fifo(&filtered);
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("Error sending audio chunk: {}", e);
                                    }
                                }
                            });
                        }
                        current_chunk.clear();
                        in_speech = false;
                        silence_samples = 0;
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    Ok(())
}

/// Create an in‑memory WAV file (as a Vec<u8>) from a slice of i16 samples.
fn create_wav_in_memory(
    samples: &[i16],
    channels: u16,
    sample_rate: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
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
    let json: serde_json::Value = response.json().await?;
    let transcription = json["text"].as_str().unwrap_or("").to_string();
    Ok(transcription)
}

/// Filter transcriptions aggressively.
/// Discards transcriptions that are too short, match unwanted phrases, or consist mostly of non-alphabetic characters.
fn filter_transcription(transcription: &str) -> Option<String> {
    let unwanted = [
        "Thank you for watching!",
        "Thank you for watching",
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございました。",
        "¡Gracias por ver!",
        "시청해주셔서 감사합니다",
        "시청해주셔서 감사합니다!",
    ];
    let trimmed = transcription.trim();
    if trimmed.is_empty() || trimmed.len() < 3 {
        return None;
    }
    if trimmed.chars().all(|c| !c.is_alphabetic()) {
        return None;
    }
    for phrase in &unwanted {
        if trimmed.eq_ignore_ascii_case(phrase) {
            return None;
        }
    }
    Some(trimmed.to_string())
}
