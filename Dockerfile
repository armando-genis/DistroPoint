# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# noninteractive APT, enable GPU + graphics capabilities
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display

# Set environment variables
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Set up timezone and locale in one layer
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    tzdata \
    locales \
    gnupg2 \
    curl \
    ca-certificates && \
    echo 'Etc/UTC' > /etc/timezone && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Verify and update NVIDIA repositories if needed
RUN rm -f /usr/share/keyrings/cuda-archive-keyring.gpg && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor --batch -o /usr/share/keyrings/cuda-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    rm -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list

# Update CUDA and cuDNN to ensure specific versions
RUN apt-get update && \
    apt-get install -y --allow-change-held-packages \
    cuda-toolkit-12-1 \
    libcudnn8 \
    libcudnn8-dev && \
    rm -rf /var/lib/apt/lists/*

# Install SO dependencies - do this BEFORE creating the user
RUN apt-get update -qq && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    libgtk-3-dev \
    pkg-config \
    iputils-ping \
    wget \
    python3-pip \
    python3-dev \
    libtool \
    libpcap-dev \
    git-all \
    libeigen3-dev \
    libpcl-dev \
    software-properties-common \
    bash-completion \
    curl \
    tmux \
    zsh \
    nano \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch, TorchVision, and Torchaudio
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required Python packages
RUN pip3 install --no-cache-dir \
    tensorboardX \
    albumentations \
    certifi \
    charset-normalizer \
    configparser==5.0.0 \
    idna \
    imageio \
    jmespath \
    joblib \
    networkx \
    numpy \
    opencv-python-headless==4.6.0.66 \
    packaging \
    Pillow \
    protobuf==3.20.1 \
    pyparsing \
    pyrr==0.10.3 \
    python-dateutil \
    PyWavelets \
    PyYAML \
    qudida \
    seaborn \
    s3transfer \
    scikit-image \
    scikit-learn \
    scipy \
    simplejson \
    six \
    threadpoolctl \
    tifffile \
    typing_extensions \
    urllib3 \
    open3d 

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Set up zsh with Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    chsh -s $(which zsh)

RUN echo 'export TERM=xterm-256color' >> ~/.zshrc && \
    echo 'alias ll="ls -alF"' >> ~/.zshrc && \
    echo 'alias la="ls -A"' >> ~/.zshrc && \
    echo 'alias l="ls -CF"' >> ~/.zshrc && \
    echo 'export ZSH_THEME="robbyrussell"' >> ~/.zshrc && \
    echo 'PROMPT="%F{yellow}%*%f %F{green}%~%f %F{blue}➜%f "' >> ~/.zshrc

# Set up tmux configuration
RUN echo 'set -g default-terminal "screen-256color"' >> ~/.tmux.conf && \
    echo 'set -g mouse on' >> ~/.tmux.conf

WORKDIR /workspace
ENV PATH="/root/.local/bin:${PATH}"