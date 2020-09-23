FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER Robert Hong <jacky1127s@gmail.com>

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install python3-dev python3-pip git libglib2.0-0 libsm6 libxext6 libxrender-dev python3-tk -y
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    x11-apps ffmpeg eog \
    build-essential cmake \
    libgtk2.0-dev \
    pkg-config \
    libopenblas-dev


#ADD ./requirements.txt /
#WORKDIR /
#RUN pip3 install -r requirements.txt

# Science libraries and other common packages
RUN pip3 --no-cache-dir install \
    numpy scipy sklearn scikit-image pandas matplotlib Cython requests

# Jupyter notebook
RUN pip3 --no-cache-dir install jupyter && \
    mkdir /root/.jupyter && \
    echo "c.NotebookApp.ip = '*'" \
         "\nc.NotebookApp.open_browser = False" \
         "\nc.NotebookApp.token = ''" \
         > /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888

# Expose port for TensorBoard
EXPOSE 6006


RUN pip3 install --upgrade pip
# opencv-python
RUN pip3 install opencv-python==3.4.11.41

#RUN pip3 install --upgrade pip
RUN pip3 install cupy-cuda101

# Apex install
#RUN git clone https://github.com/NVIDIA/apex
#WORKDIR apex/
#RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#RUN pip3 install -v --no-cache-dir ./
#WORKDIR /

# DALI install --cuda10.1
RUN pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100

# install gradual warmup lr
RUN pip3 install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git









# DL framework
# Tensorflow 1.14.0 - GPU
#RUN pip3 install tensorflow-gpu==1.14.0

# Keras 2.2.4 
#RUN pip3 install keras==2.2.4

# torch
RUN pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torchsummary
RUN pip3 install tensorboardX
RUN pip3 install crc32c

WORKDIR "/root"
CMD ["/bin/bash"]

