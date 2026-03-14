# Get Started

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python --version

poetry install
```

## Train Models

```bash
poetry run python src/app/train_intent.py
poetry run python src/app/train_scholarship.py
```

## Run Application

```bash
poetry run streamlit run src/app/demo.py
```

## Clean Up Environment

```bash
deactivate
rm -rf .venv
```