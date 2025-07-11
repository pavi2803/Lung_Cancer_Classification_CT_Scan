# Use official Python slim image (3.10 for compatibility)
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Upgrade pip and install build tools for any native packages
RUN pip install --upgrade pip setuptools wheel

# Copy requirements.txt (you'll create this next)
COPY requirements.txt .

# Install python dependencies
RUN pip install -r requirements.txt

# Copy your entire project (adjust if needed)
COPY . .

# Expose the port your FastAPI app will run on
EXPOSE 10000

# Command to run your FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]