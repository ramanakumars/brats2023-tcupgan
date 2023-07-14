# Change base Docker image if, for instance, a CUDA support is required.
FROM ubuntu:20.04
MAINTAINER Ramanakumar Sankar

# Add your project's dependencies (excluding python dependencies)
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            python3-dev \
            python3-pip\
            python3-setuptools\
            git\
            curl && \
    rm -rf /var/lib/apt/lists/*

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
