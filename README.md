# NovaAssist AI Capstone

NovaAssist is a lightweight, zero-dependency natural language intent classification assistant designed for workflow routing and query response handling.

## Project Architecture
- `src/model.py`: TF-IDF vectorization and cosine similarity classifier pipeline.
- `src/app.py`: CLI user interface and response resolution engine.
- `models/`: Serialized model artifacts and vocabulary weights.
- `tests/`: Automated unit tests using `pytest`.

## Getting Started

1. **Set up virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1