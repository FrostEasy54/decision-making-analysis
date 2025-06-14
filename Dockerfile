# Use an official Python runtime as a parent image
FROM python:alpine

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run the app via Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]