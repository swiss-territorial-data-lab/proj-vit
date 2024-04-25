FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 as base

########################################################################################################################
# Install python and pip
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y python3 python3-pip \
    && apt install -y wget git vim nano tmux rsync\
    && apt install -y htop ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# RUN mkdir /.cache && chmod 777 /.cache
########################################################################################################################

ENV CONDA_PATH=/opt/miniconda3
RUN mkdir $CONDA_PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \ 
    && bash /tmp/miniconda.sh -b -u -p $CONDA_PATH \
    && rm /tmp/miniconda.sh

ENV PATH="$CONDA_PATH/bin:$PATH"

RUN conda create -n mmseg -y python=3.9

SHELL ["conda", "run", "-n", "mmseg", "/bin/bash", "-c"]

RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

RUN conda install -c conda-forge gdal

RUN pip install -U openmim ftfy regex rasterio\
    && mim install mmengine \
    && mim install mmcv

CMD /bin/bash