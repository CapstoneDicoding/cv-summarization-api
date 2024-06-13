# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Tesseract OCR and other dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code into the container at /app
COPY . .

# Set the environment variables for Google Cloud credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/key.json"

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run the application
CMD ["python", "main.py"]
