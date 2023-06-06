FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements_cpu.txt .
RUN pip3 install -r requirements_cpu.txt

# Copy the required files into the container at /app
COPY src/ ./src
COPY conf ./conf

# Create a dir named models to store the models
RUN mkdir models

EXPOSE 5000

# Run app.py when the container launches
CMD ["python3", "src/app.py"]


