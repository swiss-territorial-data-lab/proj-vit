# Vision Transformer for Multi-Modal Remote Sensing Imagery

This repository is the implementation of STDL Vision Transformer project in collaboration with HEIG-VD. The detailed documentation of the project is available on [STDL technical website](https://tech.stdl.ch/PROJ-VIT/).  

### Table of contents

- [Requirements](#requirements)
    - [Hardware](#hardware)
    - [Software](#software)
- [Data Preparation](#data-preparation)
- [Pretraining](#pretraining)
- [Pretrained Models](#pretrained-models)
- [Fine-tuning](#finetuning)


## Requirements

### Hardware

* A CUDA-enabled GPU with at least 16GB Memory is required for experiments. 

   5 * NVIDIA A40 (48GB) is used in this project. 

### Software

* NVIDIA driver  (>=535.161.08)
* [Docker](https://www.docker.com/) (>=26.0.2)

This project was developed and tested with CUDA 12.1 on Ubuntu 22.04 Docker container. To simplify the environment configuration and version control, we distribute the docker image for reproduction. 

Note:
1. The CUDA Toolkit installed on the host machine is not relevant with the one in docker container. It is not mandatory to have CUDA Toolkit installed on host machine or to match their version. 
2. The NVIDIA driver and NVIDIA Container Toolkit must be installed on the host machine (see [this guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).

## Data Preparation

Please refer to [here](./dataset_preparation/dataset_preparation.md) for the details about dataset consolidation. The datasets are available through [STDL web service](https://data.stdl.ch/proj-vit/dataset/):

1. [RS Pretrain](https://data.stdl.ch/proj-vit/dataset/pretrain_ready/)  (491 GB) 
2. [FLAIR #1](https://data.stdl.ch/proj-vit/dataset/flair/) (96 GB)
3. [STDL-SOILS](https://data.stdl.ch/proj-vit/dataset/STDL-SOILS/) (6.4 GB) 

## Pretrain 

We forked and modified [Scale-MAE](https://github.com/bair-climate-initiative/scale-mae) to pertrain the Vision Transformer with 5-band images. All the Experiments are conducted with docker. To launch the docker container, run following command:

```
cd docker-images/scalemae_ub22.04-cuda12-cudnn8-py3.9-pytroch
docker build -t scale_mae:v0 .
docker run -it -v /mnt/:/mnt/ --gpus all --ipc=host --privileged scale_mae:v0
```

You can replace the `-v /mnt/:/mnt/` to your customized storage path that would be mounted in the container.  After successful initialization, download the pretrained model weights on fMoW dataset:

```
cd /scale-mae/mae
wget https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth
```

Then, download the dataset:

```
cd /scale-mae/mae/data 
wget -r -np -nH --cut-dirs=2 -R index.html https://data.stdl.ch/proj-vit/dataset/pretrain_ready/
```

Please make sure there is enough space in mounted volume for downloading the datasets.

To monitor the training and visualize the prediction of masked image patches, please create a [wandb](https://wandb.ai/site) account and generate a token for authentication. You can also ignore this function when the request for authentication is prompted.

To launch the pretraining, configure the number of GPUs and run:

```
cd /scale-mae/mae
conda activate scalemae

torchrun --standalone --nnodes=1 --nproc-per-node=${GPU_NUM} main_pretrain.py --config config/pretrain.yaml --epochs 800 --batch_size 32 --model mae_vit_large_patch16 --resume scalemae-vitlarge-800.pth 
``` 
This script accepts arguments for detailed training configuration, referring to [here](https://github.com/swiss-territorial-data-lab/scale-mae/blob/1db755a5b591c49f5ee983708b7244f461acd7f6/mae/main_pretrain.py#L58).
 
Once the training is finished, the pretrained weights are stored in `mae/output_dir/checkpoint-latest.pth`

## Pretrained Models

For the pretrained weights, the name of the layers might not be consistent between different implementation, e.g., Scale-MAE and MMsegmentation. We converted the key in the `state_dict` and only loaded the encoder (backbone) from these pretrained model. Here are the converted models:


* 5-band RS imagery: [**Scale-MAE pretrained model**](https://data.stdl.ch/proj-vit/weights/vit-large-scalemae-pretrained-rs-5bands.pth)
* 3-band Natural imagery: [**ImageNet-22K pretrained ViT-L Encoder**](https://data.stdl.ch/proj-vit/weights/vit-large-p16_in21k-pre-3rdparty_ft-in1k-384-mmseg.pth)


## Downstream Experiments 

Downstream semantic segmentation experiments are implemented with modified [MMsegmentation](https://github.com/swiss-territorial-data-lab/mmsegmentation) framework that supports multi-spectral data loading and augmentation. The network architecture and model training parameters can be customized in configuration file. Please refer to [official tutorial](https://mmsegmentation.readthedocs.io/en/latest/) for the usage of the framework.

Create the docker image from DockerFile and launch the container:   

```
cd docker-images/mmseg_ub22.04-cuda12-cudnn8-py3.9-pytorch
docker build -t mmseg:v0 .
docker run -it -v /mnt/:/mnt/ --gpus all --ipc=host --privileged mmseg:v0
```

In docker container, following codes initiate the environment and download the dataset:

```
conda activate mmseg
wget -r -np -nH --cut-dirs=1 https://data.stdl.ch/proj-vit/weights/

cd /mmsegmentation/data
wget -r -np -nH --cut-dirs=2 -R index.html https://data.stdl.ch/proj-vit/dataset/flair/
wget -r -np -nH --cut-dirs=2 -R index.html https://data.stdl.ch/proj-vit/dataset/STDL-SOILS/
```

### Linear Probing

To evaluate the performance of the pretrained encoder, you can choose to freeze the encoder layers and train the decoder with supervised learning on the downstream tasks. Here is an example to linear probe 5-band ViT-L encoder and UperNet decoder with (224, 224) input size on FLAIR #1 dataset:

```
cd /mmsegmentation
export PRETRAINED_WEIGHTS='weights/vit-large-scalemae-pretrained-rs-5bands.pth'
export WORK_DIR_PATH='work_dirs/linprobe-vit-l-5bands_upernet_4xb4-20k_flair-224x224'

python tools/train.py configs/vit_flair/linprobe-vit-l-5bands_upernet_4xb4-20k_flair-224x224.py --cfg-options model.backbone.init_cfg.checkpoint=${PRETRAINED_WEIGHTS} --work-dir ${WORK_DIR_PATH}
```

If training with multiple GPUs, issue the following code:

```
bash tools/dist_train.sh configs/vit_flair/linprobe-vit-l-5bands_upernet_4xb4-20k_flair-224x224.py ${GPU_NUM} --cfg-options model.backbone.init_cfg.checkpoint=${PRETRAINED_WEIGHTS} --work-dir ${WORK_DIR_PATH}
```


The checkpoints and log files are saved in `WORK_DIR_PATH`. Tensorboard is supported to visualize the training process. For example:
```
tensorboard --logdir ${WORK_DIR_PATH}/${yyyymmdd_hhmmss}/vis_data/
```
`yyyymmdd_hhmmss` denotes the start time of the training.


### Finetuning

To achieve optimal performance, both the encoder and decoder can be fine-tuned together despite linear probing. We employed finetuning on the FLAIR and SOILS dataset to explore the benefit from pretraining on large-scale remote sensing images and additional bands. To study the impact of multi-modal inputs and pretraining with remote sensing imagery, we also conducted ablation studies on FLAIR dataset.  

### STDL SOILS

Finetune the model with encoder pretrained by Scale-MAE and RS Pretrain:
```

```


### FLAIR challenge 
To conduct the ablation study Experiments, run



