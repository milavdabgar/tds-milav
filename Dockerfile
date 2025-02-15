FROM python:3.11-slim

# Install Node.js and npm
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install prettier locally
RUN mkdir -p /app/node_modules && cd /app && npm init -y && npm install prettier@3.4.2

# Copy the rest of the application
COPY . .

# Create data directory and make it writable
RUN mkdir -p /data && chmod 777 /data

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]
