# ub22.04-cuda12-cudnn8-py3.10-pytorch

Image ubuntu 22.04, cuda 12.1.1, cudnn8, avec python 3.10, pip et torch

## Build

`
docker build -t ub22.04-cuda12-cudnn8-py3.10 .
`

### Build with poetry dependency

1. Place the pyproject.toml of your project near the Dockerfile
2. Uncomment the 5 lines under `# Install dep with poetry` in the dockerfile
3. Run the docker build command

## Run

To use your gpu(s) you need: (more details in the main README.md)
1. To install the nvidia driver
2. To install the nvidia docker runtime
3. To add the option `--gpus all` when running your container


`
docker run -it --rm  --name my_docker --gpus all  ub22.04-cuda12-cudnn8-py3.10-pytorch
`