FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

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
        qt6-base-dev \
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
        nvidia-cuda-toolkit

RUN git clone https://github.com/llvm-mirror/openmp.git && \
    cd openmp && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j $(nproc) && \
    make install -j $(nproc) && \
    make PREFIX=/usr/local install -j $(nproc)

RUN git clone https://github.com/xianyi/OpenBLAS.git && \
    cd OpenBLAS && \
    mkdir build && \
    cd build && \
    cmake \
        -DUSE_OPENMP=1 \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j $(nproc) && \
    make install -j $(nproc)

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/conda/bin:$PATH
COPY environment.yml .
RUN conda env create -f environment.yml

RUN git clone https://github.com/IntelRealSense/librealsense.git && \
    cd librealsense && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=release -DCMAKE_BUILD_TYPE=release -DBUILD_PYTHON_BINDINGS:bool=true && \
    make -j $(nproc) && \
    make install

ARG USERNAME
RUN useradd -m ${USERNAME}
RUN usermod -aG video ${USERNAME}

USER ${USERNAME}

RUN mkdir -p /home/${USERNAME}/pointfusion
WORKDIR /home/${USERNAME}/pointfusion

RUN conda init bash
RUN echo "conda activate pointfusion" >> ~/.bashrc
#RUN echo "pip install -e ." >> ~/.bashrc

ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30

ENV CUDA_HOME /usr/local/cuda
ENV PATH $CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH $CUDA_HOME/lib64:$LD_LIBRARY_PATH

ENTRYPOINT [ "/bin/bash" ]