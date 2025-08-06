# 1. Base Image with Python

FROM python:3.13-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy project files
COPY . .

# 5. Expose the port and start
EXPOSE 8000

# Use Uvicorn to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]