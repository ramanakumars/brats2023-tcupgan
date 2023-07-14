# Change base Docker image if, for instance, a CUDA support is required.
FROM ubuntu:22.04
MAINTAINER Ramanakumar Sankar

# Add your project's dependencies (excluding python dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            python3-dev \
            curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Update you project dependencies in the ./requirements.txt file.
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

# A docker image must define an entry point. Rename `entry_point.py` to your main python file. Adjust if project
# contains multiple files.
COPY . /tcupgan

ENTRYPOINT ["python3", "/tcupgan/mlcube.py"]