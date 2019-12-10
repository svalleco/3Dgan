# Docker image for running TPU tensorflow examples.
FROM ubuntu:bionic

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        sudo \
        gnupg \
        lsb-release \
        ca-certificates \
        build-essential \
        git \
        python \
        python-pip \
        awscli \
        python-setuptools && \
    export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb https://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    pip install pyyaml && \
    pip install wheel && \
    pip install tensorflow==1.15.0 && \
    pip install google-cloud-storage && \
    pip install google-api-python-client && \
    pip install oauth2client



RUN git clone https://github.com/svalleco/3Dgan.git
RUN cd 3Dgan
RUN git checkout svalleco/low_precision

