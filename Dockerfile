FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y \
        wget \
        bzip2 \
        git \
        cmake \
        g++ \
        libgl1-mesa-glx \
        mlocate \
        libopenblas-base \
        libopenmpi-dev \
        libudev-dev \
        pkg-config \
        libgtk-3-dev \
        v4l-utils \
        libgl1-mesa-dri \
        mesa-utils \
        libxkbcommon-x11-0 \
        libxcb-xinerama0 \
        libglfw3-dev \
        libgl1-mesa-dev \ 
        libglu1-mesa-dev \
        libusb-1.0-0-dev \
        libssl-dev \
        libusb-1.0-0-dev \
        libudev-dev \
        pkg-config \
        libgtk-3-dev \
        mesa-utils \
        libgl1-mesa-dri \
        libgl1-mesa-glx \
        curl

RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
RUN chmod +x Mambaforge-$(uname)-$(uname -m).sh
RUN ./Mambaforge-$(uname)-$(uname -m).sh -b -p /opt/mamba

ENV PATH /opt/mamba/bin:$PATH
COPY environment.yml .
RUN mamba env create -f environment.yml

ARG USERNAME
RUN useradd -m ${USERNAME}
RUN usermod -aG video ${USERNAME}

USER ${USERNAME}

RUN mkdir -p /home/${USERNAME}/pointfusion
WORKDIR /home/${USERNAME}/pointfusion/examples

RUN conda init bash
RUN echo "conda activate pointfusion" >> ~/.bashrc

ENTRYPOINT [ "/bin/bash" ]