# Change base Docker image if, for instance, a CUDA support is required.
FROM python:3.9-slim
MAINTAINER Ramanakumar Sankar

# Add your project's dependencies (excluding python dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git
# Update you project dependencies in the ./requirements.txt file.
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

COPY . /tcupgan

WORKDIR /tcupgan

RUN git init &&\
    git remote add origin https://github.com/ramanakumars/TCuPGAN.git &&\
    git fetch &&\
    git checkout -t origin/main && git pull origin main


ENTRYPOINT ["python3", "/tcupgan/mlcube.py"]
