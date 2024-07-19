FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC \
 && apt-get -y update \
 && apt-get -y install \
    libx264-dev \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 \
    wget curl cmake build-essential pkg-config \
    libxext6 libxrender-dev ffmpeg git
    
RUN apt-get clean && rm -rf /tmp/* /var/tmp/* \
 && python3 -m pip install --upgrade pip \
 && python3 -m pip install --upgrade setuptools
# RUN conda remove --force ffmpeg -y

WORKDIR /workspace


RUN pip3 install --no-cache-dir -Iv \
    numpy==1.20.3 opencv-python==4.3.0.38 onnx==1.12.0 \
    # numpy opencv-python==4.5.5.64 onnx \
    onnxruntime-gpu==1.7.0 mxnet-cu111==1.9.1 mxnet-mkl==1.6.0 scikit-image==0.19.3 \
    # onnxruntime-gpu mxnet-cu116 mxnet-mkl scikit-image \
    insightface==0.2.0 requests==2.25.1 kornia==0.5.11 dill==0.3.7 wandb==0.16.4 \
    # insightface requests kornia dill wandb \
    # insightface requests kornia==0.5.11 dill wandb \
    notebook==6.5.6 ipython==7.29.0 ipykernel==6.16.2 psutil==5.9.2 \ 
    ##from here it is for MAE environment
    timm==0.4.12 matplotlib==3.5.3 info-nce-pytorch==0.1.4 \
    # retinaface_pytorch face-alignment
    face-alignment==1.4.1 

