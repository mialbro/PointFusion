FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

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
        libudev0 \
        pkg-config \
        libgtk-3-dev \
        mesa-utils \
        libgl1-mesa-dri \
        libgl1-mesa-glx \
        curl

RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/miniforge3

ENV PATH /opt/miniforge3/bin:$PATH
COPY environment.yml .
RUN conda update -n base -c conda-forge conda
RUN conda env create -f environment.yml

RUN apt-get install -y apt-transport-https lsb-release linux-headers-$(uname -r)

RUN mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
    tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update

RUN apt-get install -y librealsense2-utils librealsense2-dev librealsense2-dbg x11-apps mesa-utils qttools5-dev-tools usbutils
RUN apt-get update && \
    apt-get upgrade -y

ARG USERNAME
RUN useradd -m ${USERNAME}
RUN usermod -aG video ${USERNAME}

USER ${USERNAME}

RUN mkdir -p /home/${USERNAME}/pointfusion
WORKDIR /home/${USERNAME}/pointfusion

ENV PATH="/opt/miniforge3/bin:$PATH"
ENV PATH="/opt/miniforge3/envs/pointfusion/bin:${PATH}"

RUN conda init bash
RUN echo "conda activate pointfusion" >> ~/.bashrc && \
    echo "pip install -e ." >> ~/.bashrc

ENV QT_QPA_PLATFORM=xcb
ENTRYPOINT [ "/bin/bash" ]