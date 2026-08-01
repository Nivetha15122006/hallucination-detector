FROM python:3.11-slim

# Set work directory
WORKDIR /code

# Copy requirements and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create HF cache directory inside /tmp to bypass read-only filesystem issues
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

# Copy project folders
COPY ./backend /code/backend
COPY ./data /code/data

# Expose HuggingFace default port
EXPOSE 7860

# Run FastAPI server on port 7860, pointing to backend.main:app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
