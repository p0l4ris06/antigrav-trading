//! Antigravity Inference Engine — Bare-Metal Rust Binary
//!
//! Loads an ONNX policy graph exported from the Python PPO agent
//! and runs forward passes in microseconds via the `tract` runtime.
//!
//! Architecture:
//!     1. tokio TCP listener accepts observation vectors
//!     2. tract loads and optimizes the ONNX graph at startup
//!     3. Each connection receives: observation → action (allocation weights)
//!
//! This is a STUB for the colocation deployment path (Fork B).
//! Full implementation requires wiring to a market data feed.

use anyhow::{Context, Result};
use std::path::PathBuf;
use tract_onnx::prelude::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Observation dimension — must match the Python policy export.
const OBS_DIM: usize = 16;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("Antigravity Inference Engine starting...");

    // Load ONNX model
    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/policy.onnx"));

    tracing::info!("Loading ONNX model from: {:?}", model_path);

    let model = tract_onnx::onnx()
        .model_for_path(&model_path)
        .context("Failed to load ONNX model")?
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec![1, OBS_DIM as i64]),
        )?
        .into_optimized()
        .context("Failed to optimize model")?
        .into_runnable()
        .context("Failed to make model runnable")?;

    tracing::info!("Model loaded and optimized successfully");

    // Start TCP listener
    let addr = "127.0.0.1:9090";
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Listening on {}", addr);

    loop {
        let (mut socket, peer) = listener.accept().await?;
        tracing::info!("Connection from: {}", peer);

        // Clone the model for this connection
        let model = model.clone();

        tokio::spawn(async move {
            let mut buf = vec![0u8; OBS_DIM * 4]; // f32 = 4 bytes

            loop {
                // Read observation vector (OBS_DIM × f32, little-endian)
                match socket.read_exact(&mut buf).await {
                    Ok(_) => {}
                    Err(_) => break,
                }

                // Deserialize f32 values
                let obs: Vec<f32> = buf
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();

                // Run inference
                let input = tract_ndarray::arr2(&[obs])
                    .into_shape((1, OBS_DIM))
                    .unwrap();
                let input_tensor: Tensor = input.into();

                let result = match model.run(tvec![input_tensor.into()]) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("Inference failed: {}", e);
                        break;
                    }
                };

                // Extract action output
                let action = result[0]
                    .to_array_view::<f32>()
                    .expect("Failed to extract action tensor");

                // Serialize back as f32 little-endian bytes
                let mut response = Vec::new();
                for &val in action.iter() {
                    response.extend_from_slice(&val.to_le_bytes());
                }

                if socket.write_all(&response).await.is_err() {
                    break;
                }
            }

            tracing::info!("Connection closed: {}", peer);
        });
    }
}
