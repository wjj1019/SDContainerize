# Start from a base image
FROM python:3.9-slim-buster

# Setting environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    libpoppler-cpp-dev \
    libpoppler-private-dev\
    poppler-utils

# Copy project files into the docker image
COPY . /app/GaonSidedrawer
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["python", "/app/GaonSidedrawer/main.py", "--document", "/documents/document.pdf", "--model", "/documents/model.pkl", "--vectorizer", "/documents/vectorizer.pkl"]
