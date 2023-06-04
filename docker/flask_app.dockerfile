FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the required files into the container at /app
COPY src/ ./src
COPY models/UNetWithResnet50Encoder_unfreeze2_512.pth ./models/UNetWithResnet50Encoder_unfreeze2_512.pth
COPY conf ./conf

EXPOSE 5000

# Run app.py when the container launches
CMD ["python3", "src/app.py"]


