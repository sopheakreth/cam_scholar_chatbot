FROM python:3.12-slim

WORKDIR /app

# install poetry
RUN pip install --no-cache-dir poetry

# copy dependency files first
COPY pyproject.toml poetry.lock* ./

# disable virtualenv creation
RUN poetry config virtualenvs.create false

# install dependencies
#RUN poetry install --no-interaction --no-ansi --no-root
RUN poetry install --only main --no-root --no-interaction --no-ansi

# download nltk data
RUN python - <<EOF
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
EOF

# copy project files
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/app/demo.py", "--server.port=8501", "--server.address=0.0.0.0"]