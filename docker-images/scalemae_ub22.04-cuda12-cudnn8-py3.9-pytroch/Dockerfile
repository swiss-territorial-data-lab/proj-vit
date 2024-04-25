FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 as base

########################################################################################################################
# Install python and pip
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y python3 python3-pip \
    && apt install -y wget vim nano tmux\
    && apt install -y htop ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN mkdir /.cache && chmod 777 /.cache
########################################################################################################################

ENV CONDA_PATH=/opt/miniconda3
RUN mkdir $CONDA_PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \ 
    && bash /tmp/miniconda.sh -b -u -p $CONDA_PATH \
    && rm /tmp/miniconda.sh

ENV PATH="$CONDA_PATH/bin:$PATH"

RUN /opt/miniconda3/bin/conda init

RUN conda create -n scalemae -y python=3.9 geopandas rasterio shapely

SHELL ["conda", "run", "-n", "scalemae", "/bin/bash", "-c"]

RUN conda run -n scalemae /bin/bash -c conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

RUN apt-get update  \
    && apt-get upgrade -y  \
    && apt-get install  # buildkit
RUN conda install -y conda-forge::kornia gdal \
    && pip install torchgeo opencv-python classy_vision wandb tensorboard

# clone stdl scaleMAE repo
RUN git clone https://github.com/swiss-territorial-data-lab/scale-mae.git

WORKDIR "scale-mae"
RUN pip install -e .

CMD /bin/bash