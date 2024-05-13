# Vision Transformer for Multi-Modal Remote Sensing Imagery

This repository is the implementation of STDL Vision Transformer project in collaboration with HEIG-VD. The detailed documentation of the project is available on [STDL technical website](https://tech.stdl.ch/PROJ-VIT/).  

### Table of contents

- [Requirements](#requirements)
    - [Hardware](#hardware)
    - [Software](#software)
- [Data Preparation](#data-preparation)
- [Pretraining](#pretraining)
- [Pretrained Models](#pretrained-models)
- [Downstream Experiments](#downstream-experiments)
    - [Linear Probing](#linear-probing)
    - [Fine-tuning](#finetuning)
    - [Ablation Studies](#ablation-studies)
    - [Evaluation](#evaluation)


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

Please refer to [here](./dataset_preparation/README.md) for the details about dataset consolidation. The datasets are available through [STDL web service](https://data.stdl.ch/proj-vit/dataset/):

1. [RS Pretrain](https://data.stdl.ch/proj-vit/dataset/pretrain_ready/)  (491 GB) 
2. [FLAIR #1](https://data.stdl.ch/proj-vit/dataset/flair/) (96 GB)
3. [STDL-SOILS](https://data.stdl.ch/proj-vit/dataset/STDL-SOILS/) (6.4 GB) 

## Pretrain 

We forked and modified [Scale-MAE](https://github.com/bair-climate-initiative/scale-mae) to pertrain the Vision Transformer with 5-band images (RGB, NIR and nDSM). The pretraining was implemented with docker. To launch the docker container, run following command:

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

Then, download the *RS Pretrain* dataset in the mounted volume and point the dataset to project folder with soft symbolic link:

```
cd ${DATA_PATH} 
wget -r -np -nH --cut-dirs=2 -e robots=off -b https://data.stdl.ch/proj-vit/dataset/pretrain/

ln -s ${DATA_PATH}/pretrain /scale-mae/mae/data/pretrain
```

The downloading runs in back-end with `-b`. You can check the progress with `tail -f wget-log`. Please make sure there is enough space in mounted volume for downloading the datasets.

To monitor the training and visualize the prediction of masked image patches, please create a [wandb](https://wandb.ai/site) account and generate a token for authentication. You can also ignore this function when the request for authentication is prompted.

To launch the pretraining, configure the number of GPUs and run:

```
cd /scale-mae/mae
conda activate scalemae

torchrun --standalone --nnodes=1 --nproc-per-node=${GPU_NUM} main_pretrain.py --config config/pretrain.yaml --epochs 800 --batch_size 32 --model mae_vit_large_patch16 --resume scalemae-vitlarge-800.pth 
``` 
This script accepts arguments for detailed training configuration, referring to [here](https://github.com/swiss-territorial-data-lab/scale-mae/blob/1db755a5b591c49f5ee983708b7244f461acd7f6/mae/main_pretrain.py#L58).
 
Once the training terminates, the pretrained weights are stored in `mae/output_dir/checkpoint-latest.pth`

## Pretrained Models

For the pretrained weights, the name of the layers might not be consistent between different implementation, e.g., Scale-MAE and MMsegmentation. We converted the key in the `state_dict` and only loaded the encoder (backbone) from these pretrained model. Here are the converted models:


* 5-band Remote Sensing imagery: [**Scale-MAE pretrained model**](https://data.stdl.ch/proj-vit/weights/vit-large-scalemae-pretrained-rs-5bands.pth)
* 3-band Natural imagery: [**ImageNet-22K pretrained ViT-L Encoder**](https://data.stdl.ch/proj-vit/weights/vit-large-p16_in21k-pre-3rdparty_ft-in1k-384-mmseg.pth)

*Note: the 3-band Natural imagery pretrained encoder is first training with self-supervised learning (MIM) on ImageNet-21K and further trained with supervised learning (scene classification) on ImageNet-1K. See the [reference](https://github.com/open-mmlab/mmpretrain/tree/main/configs/vision_transformer).*

## Downstream Experiments 

Downstream semantic segmentation experiments are implemented with modified [MMsegmentation](https://github.com/swiss-territorial-data-lab/mmsegmentation) framework that supports multi-spectral data loading and augmentation. The network architecture and model training parameters can be customized in configuration file. Please refer to [official tutorial](https://mmsegmentation.readthedocs.io/en/latest/) for the usage of the framework.

Create the docker image from DockerFile and launch the container:   

```
cd docker-images/mmseg_ub22.04-cuda12-cudnn8-py3.9-pytorch
docker build -t mmseg:v0 .
docker run -it -v /mnt/:/mnt/ --gpus all --ipc=host --privileged mmseg:v0
```

In docker container, following codes initiate the environment and download the dataset / model weights:

```
conda activate mmseg
wget -r -np -nH -e robots=off --cut-dirs=1 https://data.stdl.ch/proj-vit/weights/

cd ${DATA_PATH} 
wget -r -np -nH -e robots=off --cut-dirs=2 -b https://data.stdl.ch/proj-vit/dataset/flair/
wget -r -np -nH -e robots=off --cut-dirs=2 -b https://data.stdl.ch/proj-vit/dataset/STDL-SOILS/

ln -s ${DATA_PATH}/flair /mmsegmentation/data/flair
ln -s ${DATA_PATH}/STDL-SOILS /mmsegmentation/data/STDL-SOILS
```
The downloading runs in back-end with `-b`. You can check the progress with `tail -f wget-log`. Please make sure there is enough space in mounted volume for downloading the datasets.

### Linear Probing

To evaluate the performance of the pretrained encoder, you can choose to freeze the encoder layers and train the decoder with supervised learning on the downstream tasks. Here is an example to linear probe 5-band ViT-L encoder and UperNet decoder with (224, 224) input size on FLAIR #1 dataset:

```
cd /mmsegmentation
export PRETRAINED_WEIGHTS='weights/vit-large-scalemae-pretrained-rs-5bands.pth'
export WORK_DIR_PATH='work_dirs/linprobe-vit-l-5bands_upernet_4xb4-20k_flair-224x224'

python tools/train.py configs/vit_flair/vit-l-5bands_upernet_4xb4-160k_flair-224x224.py --cfg-options model.backbone.init_cfg.checkpoint=${PRETRAINED_WEIGHTS} model.backbone.frozen_exclude=[] --work-dir ${WORK_DIR_PATH}
```

If training with multiple GPUs, issue the following code:

```
bash tools/dist_train.sh configs/vit_flair/vit-l-5bands_upernet_4xb4-160k_flair-224x224.py ${GPU_NUM} --cfg-options model.backbone.init_cfg.checkpoint=${PRETRAINED_WEIGHTS} model.backbone.frozen_exclude=[] --work-dir ${WORK_DIR_PATH}
```


The checkpoints and log files are saved in `WORK_DIR_PATH`. Tensorboard is supported to visualize the training process. For example:
```
tensorboard --logdir ${WORK_DIR_PATH}/${yyyymmdd_hhmmss}/vis_data/
```
`yyyymmdd_hhmmss` denotes the start time of the training.


### Finetuning

To achieve optimal performance, both the encoder and decoder can be fine-tuned together despite linear probing. We employed finetuning on the FLAIR and SOILS dataset to explore the benefit from pretraining on large-scale remote sensing images and additional bands. 

Finetune the 5-band model with encoder pretrained by Scale-MAE and *RS Pretrain* dataset:

#### STDL SOILS

```
export PRETRAINED_WEIGHTS='weights/vit-large-scalemae-pretrained-rs-5bands.pth'
export WORK_DIR_PATH='work_dirs/finetune_vit-l16-ln_mln_upernet_4xb4-160k_soils-512x512'

python tools/train.py configs/stdl_soils/finetune_vit-l_upernet_4xb4-160k_soils-512x512.py --cfg-options model.backbone.init_cfg.type='Pretrained' model.backbone.init_cfg.checkpoint=${PRETRAINED_WEIGHTS} --work-dir ${WORK_DIR_PATH}
```

#### FLAIR #1 

Model with small window size (224x224):
```
export PRETRAINED_WEIGHTS='weights/vit-large-scalemae-pretrained-rs-5bands.pth'
export WORK_DIR_PATH='work_dirs/finetune_vit-l-5bands_upernet_4xb4-160k_flair-224x224'

python tools/train.py configs/vit_flair/vit-l-5bands_upernet_4xb4-160k_flair-224x224.py --cfg-options model.backbone.init_cfg.type='Pretrained' model.backbone.init_cfg.checkpoint=${PRETRAINED_WEIGHTS} --work-dir ${WORK_DIR_PATH}
```

Model with large window size (512x512):
```
export PRETRAINED_WEIGHTS='weights/vit-large-scalemae-pretrained-rs-5bands.pth'
export WORK_DIR_PATH='work_dirs/finetune_vit-l-5bands_upernet_4xb4-160k_flair-512x512'

python tools/train.py configs/vit_flair/vit-l-5bands_upernet_4xb4-160k_flair-base.py --cfg-options model.backbone.init_cfg.type='Pretrained' model.backbone.init_cfg.checkpoint=${PRETRAINED_WEIGHTS} --work-dir ${WORK_DIR_PATH}
```

Model with 5-band Photometric Distortion augmentation (Brightness / Contrast Scaling / Gaussine Noise):
```
export PRETRAINED_WEIGHTS='weights/vit-large-scalemae-pretrained-rs-5bands.pth'
export WORK_DIR_PATH='work_dirs/PhotoAug_vit-l-5bands_upernet_4xb4-160k_flair-512x512'

python tools/train.py configs/vit_flair/aug_vit-l-5bands_upernet_4xb4-160k_flair-512x512.py --cfg-options model.backbone.init_cfg.type='Pretrained' model.backbone.init_cfg.checkpoint=${PRETRAINED_WEIGHTS} --work-dir ${WORK_DIR_PATH}
```

### Ablation Studies

To study the impact of multi-modal inputs and pretraining with remote sensing imagery respectively, we conducted ablation studies on FLAIR dataset. Two base template file are configured for [3-band](https://github.com/swiss-territorial-data-lab/mmsegmentation/blob/main/configs/vit_flair/vit-l-3bands_upernet_4xb4-160k_flair-base.py) and [5-band](https://github.com/swiss-territorial-data-lab/mmsegmentation/blob/main/configs/vit_flair/vit-l-5bands_upernet_4xb4-160k_flair-base.py) model.  

The baseline model is trained from scratch with a truncated normal initialization:

```
# 3-band scratch initialization 
export WORK_DIR_PATH='work_dirs/ablation-3bands-scratch-init'
python tools/train.py configs/vit_flair/vit-l-3bands_upernet_4xb4-160k_flair-base.py --work-dir ${WORK_DIR_PATH}

# 5-band scratch initialization 
export WORK_DIR_PATH='work_dirs/ablation-5bands-scratch-init'
python tools/train.py configs/vit_flair/vit-l-5bands_upernet_4xb4-160k_flair-base.py --work-dir ${WORK_DIR_PATH}
```

Models with encoder pretrained by 5-band remote sensing imagery:

```
export PRETRAINED_WEIGHTS='weights/vit-large-scalemae-pretrained-rs-5bands.pth'

# 3-band RS pretrained 
export WORK_DIR_PATH='work_dirs/ablation-3bands-rs-init'
python tools/train.py configs/vit_flair/vit-l-3bands_upernet_4xb4-160k_flair-base.py --cfg-options model.backbone.init_cfg.type='Pretrained_Part' model.backbone.init_cfg.checkpoint=${PRETRAINED_WEIGHTS} --work-dir ${WORK_DIR_PATH}

# 5-band RS pretrained 
export WORK_DIR_PATH='work_dirs/ablation-5bands-rs-init'
python tools/train.py configs/vit_flair/vit-l-5bands_upernet_4xb4-160k_flair-base.py --cfg-options model.backbone.init_cfg.type='Pretrained_Part' model.backbone.init_cfg.checkpoint=${PRETRAINED_WEIGHTS} --work-dir ${WORK_DIR_PATH}
```
*Note: when loading the 5-band pretrained encoder weights to 3-band model, only weights for RGB channels are used.*

Models with encoder pretrained by 3-band natural imagery:

```
export PRETRAINED_WEIGHTS='weights/vit-large-p16_in21k-pre-3rdparty_ft-in1k-384-mmseg.pth'

# 3-band natural pretrained 
export WORK_DIR_PATH='work_dirs/ablation-3bands-natural-init'
python tools/train.py configs/vit_flair/vit-l-3bands_upernet_4xb4-160k_flair-base.py --cfg-options model.backbone.init_cfg.type='Pretrained_Part' model.backbone.init_cfg.checkpoint=${PRETRAINED_WEIGHTS} --work-dir ${WORK_DIR_PATH}

# 5-band natural pretrained 
export WORK_DIR_PATH='work_dirs/ablation-5bands-natural-init'
python tools/train.py configs/vit_flair/vit-l-5bands_upernet_4xb4-160k_flair-base.py --cfg-options model.backbone.init_cfg.type='Pretrained_Part' model.backbone.init_cfg.checkpoint=${PRETRAINED_WEIGHTS} --work-dir ${WORK_DIR_PATH}
```
*Note: when loading the 3-band pretrained encoder weights to 5-band model, random initialization is deployed on the two additional bands! If you want to copy the weigts from optical band, add an extra argment `model.backbone.init_cfg.copy_rgb=True` after `--cfg-options`.*


### Evaluation 

During linear probing / finetuning, the two latest checkpoints and the checkpoint with best mIoU on validation set is saved in the `WORK_DIR_PATH`. We publish all the finetuned model weights, which you can find in folder `weights` with previous wget downloading. To evaluate the model performance on the test set, following code launches inferencing:

```
# Testing on a single GPU
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# Testing on multiple GPUs
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
```

This tool accepts several optional arguments, including:

--`work-dir`: If specified, results will be saved in this directory. If not specified, the results will be automatically saved to work_dirs/{CONFIG_NAME}.

--`cfg-options`: If specified, the key-value pair in xxx=yyy format will be merged into the config file.


Here is an example on FLAIR #1 dataset:
```
export CONFIG_FILE='configs/vit_flair/vit-l-5bands_upernet_4xb4-160k_flair-base.py'
export CHECKPOINT_FILE='weights/rs-pt-vit-l-5bands_upernet_4xb4-flair-512x512.pth'
export WORK_DIR_PATH='work_dirs/finetune_vit-l-5bands_upernet_4xb4-160k_flair-512x512'

# Testing on a single GPU
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --work-dir ${WORK_DIR_PATH}

# Testing on multiple GPUs
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}
```