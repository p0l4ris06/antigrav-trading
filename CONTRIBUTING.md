# Contributing to ANTIGRAV TRADING

Thank you for your interest in contributing to **ANTIGRAV TRADING**!

## Development Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/p0l4ris06/antigrav-trading.git
   cd antigrav-trading
   ```

2. **Set Up Python Environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

3. **Run Unit Tests**:
   ```bash
   pytest -v tests/
   ```

4. **Run Server & Dashboard**:
   ```bash
   uv run antigrav serve
   ```

## Pull Request Guidelines

- Ensure unit tests pass (`pytest -v tests/`).
- Document significant quantitative strategy changes in `CHANGELOG.md`.
- Keep pull requests focused on a single feature or bugfix.
