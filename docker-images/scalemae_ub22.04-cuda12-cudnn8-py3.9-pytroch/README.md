# scalemae_ub22.04-cuda12-cudnn8-py3.9-pytroch

Docker image built on Unbuntu 22.04, CUDA 12.1, Python 3.9 and Pytorch for [scaleMAE](https://github.com/swiss-territorial-data-lab/scale-mae/tree/main)  

## Build

`
docker build -t scale_mae:v0 .
`

## Run

`
docker run -it -v /mnt/:/mnt/ --gpus all --ipc=host --privileged scale_mae:v0
`

 **-v**: mount the storage on host machine to the docker contrainer

 **--gpus**: assign GPU access to container

    e.g. --gpus '"device=0,1,2,3"'

 **--ipc**: share host memory between containers 

 **--privileged**: grants root capabilities to all devices on the host system.

    e.g. Necessary when mount S3 bucket