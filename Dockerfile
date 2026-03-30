FROM python:3.9

# Create a user to avoid running as root (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the rest of your files
COPY --chown=user . /app

# Run your Flask app on port 7860
CMD ["python", "flask_app.py"]
