FROM python:3.9-slim-buster

ARG USER=app
ARG ID=1000
ARG HOME_DIR="/home/$USER"
ARG REQUIREMENTS_TXT="requirements_cpu.txt"

RUN groupadd -g $ID $USER && useradd -g $ID -m -u $ID -s /bin/bash $USER
# Set the working directory to /app
WORKDIR $HOME_DIR
USER $USER

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY --chown=$ID:$ID REQUIREMENTS_TXT .
RUN pip3 install -r REQUIREMENTS_TXT

# Copy the required files into the container at /app
COPY --chown=$ID:$ID src/ ./src
COPY --chown=$ID:$ID conf/ ./conf

# Create a dir named models to store the models
RUN mkdir models

EXPOSE 5000

# Run app.py when the container launches
CMD ["python3", "src/app.py"]


