# Pix2PixHD
Unofficial Pytorch implementation of Pix2PixHD, from [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/abs/1711.11585) (Wang et al. 2018). Implementation for [Generative Adversarial Networks (GANs) Specialization](https://www.coursera.org/specializations/generative-adversarial-networks-gans) course material.

## Usage
1. Download the [Cityscapes dataset](https://www.cityscapes-dataset.com/), unzip the `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip` folders and move them to `data` directory.
2. All Python requirements can be found in `requirements.txt`. Support for Python>=3.7.
3. Configs for low- and high-resolution training can be found in the `configs` folder. All defaults are as per the configurations described in the original paper and code.

### Training
By default, all checkpoints will be stored in `logs/YYYY-MM-DD_hh_mm_ss`, but this can be edited via the `train.log_dir` field in the config files.

1. To train low-resolution models, run `python train.py --config configs/lowres.yml`.
2. To train high-resolution models, edit the `pretrain_checkpoint` field in `configs/highres.yml` to reflect the desired pretrained checkpoints from `2.` and ryn `python train.py --config configs/highres.yml --high_res`.

### Inference
1. Edit the `resume_checkpoint` field `configs/highres.yml` to reflect the desired high-res checkpoint from training and run `python infer.py --config configs/highres.yml`.
