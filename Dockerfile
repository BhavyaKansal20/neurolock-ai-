FROM python:3.10-slim

# System deps for OpenCV + dlib (face-recognition)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev \
    libx11-dev libgtk-3-dev libboost-python-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create needed directories
RUN mkdir -p data/faces data/face_detector exports/reports checkpoints logs

EXPOSE 5000

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "5000"]
