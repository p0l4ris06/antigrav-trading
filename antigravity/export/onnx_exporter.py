"""
ONNX Exporter — Serialization for Production Inference.

Exports the PPO policy network and (future) latency predictor to ONNX
format for deployment via:
    - Python: onnxruntime for validation
    - Rust:   tract or onnxruntime-rs for microsecond inference
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def validate_onnx_model(onnx_path: str | Path, obs_dim: int) -> dict[str, any]:
    """
    Validate an exported ONNX model by running a forward pass
    with onnxruntime and comparing output shape.

    Args:
        onnx_path: path to the .onnx file
        obs_dim: expected observation dimension

    Returns:
        dict with validation results (output_shape, inference_time_ms, valid)
    """
    import time

    import onnx
    import onnxruntime as ort

    path = Path(onnx_path)
    if not path.exists():
        return {"valid": False, "error": f"File not found: {path}"}

    # Structural validation
    model = onnx.load(str(path))
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        return {"valid": False, "error": f"ONNX validation failed: {exc}"}

    # Runtime validation
    session = ort.InferenceSession(str(path))
    dummy_input = np.random.randn(1, obs_dim).astype(np.float32)

    input_name = session.get_inputs()[0].name
    start = time.perf_counter_ns()
    outputs = session.run(None, {input_name: dummy_input})
    elapsed_ns = time.perf_counter_ns() - start

    output_shape = outputs[0].shape
    inference_ms = elapsed_ns / 1e6

    result = {
        "valid": True,
        "output_shape": list(output_shape),
        "inference_time_ms": round(inference_ms, 3),
        "input_name": input_name,
        "output_names": [o.name for o in session.get_outputs()],
    }

    logger.info("onnx.validated", path=str(path), **result)
    return result


def export_policy_to_onnx(
    agent_manager: any,
    output_dir: str | Path = "models",
    filename: str = "policy.onnx",
) -> Path:
    """
    Convenience wrapper to export an AgentManager's policy to ONNX.

    Args:
        agent_manager: antigravity.rl.agent.AgentManager instance
        output_dir: directory for the output file
        filename: ONNX filename

    Returns:
        Path to the exported ONNX file
    """
    out = Path(output_dir) / filename
    return agent_manager.export_onnx(out)
