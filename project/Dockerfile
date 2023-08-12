# Change base Docker image if, for instance, a CUDA support is required.
FROM python:3.9-slim
MAINTAINER Ramanakumar Sankar

RUN apt-get update && \
    apt-get install -y --no-install-recommends git

COPY . /tcupgan

WORKDIR /tcupgan

RUN git init &&\
    git remote add origin https://github.com/ramanakumars/TCuPGAN.git &&\
    git fetch &&\
    git checkout -t origin/main && git pull origin main

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -U -r requirements-brats.txt

ENTRYPOINT ["python3", "/tcupgan/mlcube.py"]
