# Use a Python image with PyTorch pre-installed
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy your files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Expose the port Flask will run on
EXPOSE 7860

# Command to run your app (Hugging Face expects port 7860)
CMD ["python", "flask_app.py"]
