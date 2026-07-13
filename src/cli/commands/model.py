import click
import os
import sys

@click.group()
def model():
    """Model management commands (list, inspect, export)."""
    pass

@model.command("list")
def list_models():
    """List all available models in the registry."""
    model_dir = os.getenv("AG_MODEL_DIR", "models")
    if not os.path.exists(model_dir):
        click.echo(f"Model directory '{model_dir}' does not exist.")
        return
    
    click.echo(f"Models in '{model_dir}':")
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".zip") or file.endswith(".pkl") or file.endswith(".onnx"):
                rel_path = os.path.relpath(os.path.join(root, file), model_dir)
                click.echo(f" - {rel_path}")

@model.command("inspect")
@click.option("--path", "-p", required=True, help="Path to the model zip file")
def inspect_model(path):
    """Inspect model observation and action spaces."""
    from stable_baselines3 import PPO
    
    if not os.path.exists(path):
        if os.path.exists(path + ".zip"):
            path = path + ".zip"
        else:
            click.echo(f"Model file '{path}' not found.")
            sys.exit(1)
            
    try:
        model = PPO.load(path, device="cpu")
        click.echo(f"Model: {path}")
        click.echo(f"Observation space: {model.observation_space}")
        click.echo(f"Action space: {model.action_space}")
        click.echo(f"Policy: {model.policy.__class__.__name__}")
    except Exception as e:
        click.echo(f"Error loading model: {e}")
        sys.exit(1)

@model.command("export")
@click.option("--input", "-i", required=True, help="Path to input stable-baselines3 model zip")
@click.option("--output", "-o", required=True, help="Path to save output ONNX model")
def export_model(input, output):
    """Export a trained model policy to ONNX format."""
    import torch
    from stable_baselines3 import PPO
    
    if not os.path.exists(input):
        if os.path.exists(input + ".zip"):
            input = input + ".zip"
        else:
            click.echo(f"Input model '{input}' not found.")
            sys.exit(1)
            
    click.echo(f"Loading model '{input}'...")
    try:
        model = PPO.load(input, device="cpu")
        policy = model.policy
        policy.eval()
        
        class ONNXPolicyWrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy
                
            def forward(self, obs):
                features = self.policy.features_extractor(obs)
                latent_pi, _ = self.policy.mlp_extractor(features)
                mean_actions = self.policy.action_net(latent_pi)
                return mean_actions

        wrapper = ONNXPolicyWrapper(policy)
        dummy_input = torch.randn(1, *model.observation_space.shape)
        
        if os.path.dirname(output):
            os.makedirs(os.path.dirname(output), exist_ok=True)
        click.echo(f"Exporting policy to ONNX format at '{output}'...")
        torch.onnx.export(
            wrapper,
            dummy_input,
            output,
            opset_version=12,
            input_names=["input"],
            output_names=["output"]
        )
        click.echo("Export successful.")
    except Exception as e:
        click.echo(f"Export failed: {e}")
        sys.exit(1)
