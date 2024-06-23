# hash:sha256:aca5638064de7676ce51b5d06114bf9495737fc1bb1561720296ec07ecf4be46
# hash:sha256:1358ae631cea9e003084cc9c733c5879b67872978be4e594a41dba1219af31d9
FROM registry.codeocean.com/codeocean/ubuntu:20.04-cuda11.7.0-cudnn8

ENV DEBIAN_FRONTEND=noninteractive

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
curl \
ca-certificates \
sudo \
git \
bzip2 \
libx11-6 \
build-essential \
wget \
manpages-dev \
g++ \
gcc \
libssl-dev \
unzip \
libidn11-dev \
build-essential \
cmake \
python-setuptools \
&& rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
apt-get install -y software-properties-common && \
add-apt-repository -y ppa:deadsnakes/ppa && \
apt-get update && \
apt install -y python3.10 && \
apt-get install -y python3.10-dev

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
cmake \
g++ \
python3-dev \
python3-pip \
libpython3-dev \
openssl \
jupyter-core

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py &&  python3.10 get-pip.py && rm get-pip.py

RUN apt-get update && apt-get install -y python3-pip
RUN pip install setuptools==69.0.3

RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip --no-cache-dir install \
pandas \
more-itertools \
llvmlite \
cornac==1.16.0 \
numpy==1.24.3 \
scipy==1.10.1

RUN pip --no-cache-dir install cython==0.29.28

RUN pip --no-cache-dir install --no-cache-dir jupyterlab jupyter

COPY postInstall /
RUN /postInstall
