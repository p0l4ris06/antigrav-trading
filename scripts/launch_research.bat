@echo off
echo Starting autoresearcher (50 iterations, multi-asset, 50k timesteps)...
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    call venv\Scripts\activate.bat
)
set AZURE_OPENAI_ENDPOINT=https://obsidian-zettelkasten.openai.azure.com/
python auto_optimizer.py --provider azure --model gpt-5.4-mini --iterations 50 --patience 8 --resume
