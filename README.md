# Diffusion Toy Example: CelebA Text-Conditioned & Latent Diffusion

This project demonstrates training and sampling of text-conditioned diffusion models and latent diffusion models (LDM) on the CelebA face dataset. It includes scripts for data preparation, training, and image generation.

## Features

- Prepare CelebA dataset and convert attributes to text captions
- Train pixel-space and latent-space diffusion models (LDM)
- Multi-GPU and torchrun (DDP) support
- Resume training from checkpoints
- Sample images from trained LDM using text prompts
- Live visualization of denoised vs. ground truth images during training

## Folder Structure

```
prepare_celeba.py           # Data preparation and PyTorch dataset
train_diffusion_celeba.py   # Pixel-space diffusion training
train_latent_diffusion_celeba.py # Latent diffusion training (LDM)
sample_latent_diffusion_celeba.py # Generate images from text prompts
checkpoints/                # Model checkpoints
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yunjinli/diffusion_toy_example.git
cd diffusion_toy_example
```

### 2. Create and activate a conda environment

```bash
conda create -n celeba_diffusion python=3.9 -y
conda activate celeba_diffusion
```

### 3. Install required libraries

```bash
# PyTorch (choose the right CUDA version for your GPU)
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -y

# Hugging Face Diffusers & Transformers
pip install diffusers transformers

# Other dependencies
pip install tqdm opencv-python pandas pillow matplotlib
```

### 4. (Optional) For multi-GPU training

```bash
pip install torch --upgrade
```

## Data Preparation

1. Download `img_align_celeba.zip` and `list_attr_celeba.txt` from the [official CelebA site](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
2. Place them in `/mnt/sda/celeba` (or update `DATA_DIR` in `prepare_celeba.py`).
3. Run:

```bash
python prepare_celeba.py
```

## Training

### Pixel-space diffusion

```bash
python train_diffusion_celeba.py
```

### Latent diffusion (LDM)

Single GPU:

```bash
python train_latent_diffusion_celeba.py
```

Multi-GPU (DDP):

```bash
torchrun --nproc_per_node=2 train_latent_diffusion_celeba.py
```

Resume training:

```bash
RESUME_CHECKPOINT=checkpoints/ldm_epoch_5.pt python train_latent_diffusion_celeba.py
```

## Sampling

Generate images from text prompts using a trained LDM:

```bash
python sample_latent_diffusion_celeba.py
```

## Notes

- Checkpoints are saved in `checkpoints/` after each epoch.
- The text encoder (CLIP) is frozen during training for best results.
- Visualization panels show denoised and ground truth images during training.
- For custom data paths or batch sizes, edit the relevant script variables.

## License

MIT
