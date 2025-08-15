import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from prepare_celeba import CelebATextDataset, IMG_DIR, ATTR_PATH
from tqdm import tqdm
import cv2
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary

from distributed import (
    setup_distributed, 
    cleanup_distributed, 
    is_main_process,
    get_rank,
    get_world_size,
    init_seeds
)

# import lpips

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Hyperparameters
# BATCH_SIZE = 128
BATCH_SIZE = (int)(os.environ.get('BATCH_SIZE', 128))
IMG_SIZE = 128
# EPOCHS = 10
EPOCHS = (int)(os.environ.get('EPOCHS', 10))
LR = 1e-4


DEVICE, rank, world_size, local_rank = setup_distributed()

try:
    # Distributed training setup
    use_ddp = False
    # local_rank = 0
    if 'RANK' in os.environ:
        # dist.init_process_group(backend='nccl')
        # local_rank = int(os.environ['LOCAL_RANK'])
        # torch.cuda.set_device(local_rank)
        # DEVICE = f'cuda:{local_rank}'
        use_ddp = True
    else:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not use_ddp or dist.get_rank() == 0:
        print("Training Setting: ")
        print(f"BATCH_SIZE: ", BATCH_SIZE)
        print(f"EPOCHS: ", EPOCHS)

    # Load VAE (pretrained)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae = vae.to(DEVICE)
    # if use_ddp:
    #     vae = DDP(vae, device_ids=[local_rank])
    vae.eval()  # VAE is frozen during LDM training

    # if not use_ddp or dist.get_rank() == 0:
    #     print(summary(vae))
    # Load text encoder and tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    text_encoder.eval()
    # Freeze text encoder
    for param in text_encoder.parameters():
        param.requires_grad = False

    # print(summary(text_encoder))

    # Load UNet and scheduler (smaller config for speed)
    unet = UNet2DConditionModel(
        sample_size=16,  # Latent size (VAE downsamples by 4)
        in_channels=4,   # VAE latent channels
        out_channels=4,
        layers_per_block=2,
        # block_out_channels=(128, 256, 512),
        # block_out_channels=(64, 128, 256),
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D"
        ),
        cross_attention_dim=text_encoder.config.hidden_size,
    ).to(DEVICE)

    if is_main_process():
        print(summary(unet))

    if use_ddp:
        unet = DDP(unet, device_ids=[local_rank])
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Load CelebA dataset
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1] for VAE
    ])
    dataset = CelebATextDataset(IMG_DIR, ATTR_PATH, transform=transform)
    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

    # Checkpoint directory
    checkpoint_dir = "checkpoints1"

    if not use_ddp or dist.get_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume from checkpoint if specified
    resume_path = os.environ.get('RESUME_CHECKPOINT', None)
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        unet_state = checkpoint['unet_state_dict']
        if use_ddp:
            unet.module.load_state_dict(unet_state)
        else:
            unet.load_state_dict(unet_state)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed at epoch {start_epoch}")

    best_loss = float('inf')
    # perceptual_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE) # closer to "traditional" perceptual loss, when used for optimization
    
    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        loss_meter = AverageMeter()
        # batch_time = AverageMeter()
        # data_time = AverageMeter()
        cuda_mem = AverageMeter()
        
        if use_ddp:
            sampler.set_epoch(epoch)
        if is_main_process():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        else:
            pbar = dataloader

        for i, (captions, images) in enumerate(pbar):
            images = images.to(DEVICE)
            # Encode images to latent space
            with torch.no_grad():
                # if use_ddp:
                #     latents = vae.module.encode(images).latent_dist.sample() * 0.18215  # SD scaling
                # else:
                latents = vae.encode(images).latent_dist.sample() * 0.18215  # SD scaling
                    
            # Tokenize and encode text
            # inputs = tokenizer(list(captions), padding="max_length", max_length=77, return_tensors="pt")
            inputs = tokenizer(list(captions), padding="max_length", truncation=True, max_length=77, return_tensors="pt")
            input_ids = inputs.input_ids.to(DEVICE)
            attention_mask = inputs.attention_mask.to(DEVICE)
            text_embeds = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            # Sample random timesteps
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE)
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            # Predict noise in latent space
            noise_pred = unet(noisy_latents, timesteps, text_embeds).sample
            # Loss
            # loss = torch.nn.functional.mse_loss(noise_pred, noise) / (latents.var() + 1e-5)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # denoised_latents = (noisy_latents - noise_pred).clamp(-1, 1)
            # with torch.no_grad():
            #     denoised_imgs = vae.decode(denoised_latents / 0.18215).sample.clamp(-1, 1)
            
            # perceptual_loss = perceptual_loss_fn(denoised_imgs, images).mean()
            # loss += perceptual_loss * 0.1  # Adjust weight as needed
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # pbar.set_postfix({"loss": loss.item()})

            loss_meter.update(loss.item(), images.size(0))
            if torch.cuda.is_available():
                cuda_mem.update(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024))
            if is_main_process():
                monitor = {
                    'Loss': f'{loss_meter.val:.4f} ({loss_meter.avg:.4f})',
                    'CUDA': f'{cuda_mem.val:.2f} ({cuda_mem.avg:.2f})',
                    # 'PLoss': f'{perceptual_loss.item():.4f}',
                }
                pbar.set_postfix(monitor)
            # Visualize denoised vs. ground truth every 5 steps
            if not use_ddp or dist.get_rank() == 0:
                if i % 10 == 0:
                    with torch.no_grad():
                        # Denoise latents (simple subtraction)
                        # denoised_latents = (noisy_latents - noise_pred).clamp(-1, 1)
                        alpha_bar = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)  # (B,1,1,1)
                        pred_x0 = (noisy_latents - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
                        # Decode to image space
                        # if use_ddp:
                        #     denoised_imgs = vae.module.decode(denoised_latents / 0.18215).sample.clamp(0, 1).cpu()
                        # else:
                        # denoised_imgs = vae.decode(denoised_latents / 0.18215).sample.clamp(0, 1).cpu()
                        # denoised_imgs = vae.decode(denoised_latents / 0.18215).sample.clamp(-1, 1).cpu()
                        with torch.no_grad():
                            denoised_imgs = vae.decode(pred_x0 / 0.18215).sample
                        gt_imgs = images.cpu()
                        gt_imgs = (gt_imgs * 0.5 + 0.5).clamp(0, 1)

                        denoised_imgs = (denoised_imgs * 0.5 + 0.5).clamp(0, 1).cpu()
                        panels = []
                        for idx in range(denoised_imgs.shape[0]):
                            denoised = denoised_imgs[idx].permute(1, 2, 0).numpy()
                            gt = gt_imgs[idx].permute(1, 2, 0).numpy()
                            denoised = np.clip(denoised, 0, 1)
                            gt = np.clip(gt, 0, 1)
                            denoised_bgr = cv2.cvtColor((denoised * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                            gt_bgr = cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                            panels.append(cv2.hconcat([denoised_bgr, gt_bgr]))
                        # Arrange panels in a grid: max 4 rows, more columns if needed
                        max_rows = 4
                        batch = len(panels)
                        n_cols = int(np.ceil(batch / max_rows))
                        grid = []
                        for r in range(max_rows):
                            row_panels = panels[r*n_cols:(r+1)*n_cols]
                            if row_panels:
                                # Pad row if not enough columns
                                while len(row_panels) < n_cols:
                                    h, w, c = row_panels[0].shape
                                    row_panels.append(np.zeros((h, w, c), dtype=np.uint8))
                                grid.append(cv2.hconcat(row_panels))
                        panel = cv2.vconcat(grid)
                        cv2.imshow('Denoised | Ground Truth', panel)
                        cv2.waitKey(1)
        if is_main_process():
            print(f"Epoch {epoch+1} finished.")
            if loss_meter.avg < best_loss:
                print(f"Saved best model at epoch {epoch + 1}")
                best_loss = loss_meter.avg
                unet_state = unet.module.state_dict() if use_ddp else unet.state_dict()
                torch.save({
                    'epoch': epoch + 1,
                    'unet_state_dict': unet_state,
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(checkpoint_dir, f"ldm_epoch_best.pt"))
        # Save checkpoint only on rank 0
        if not use_ddp or dist.get_rank() == 0:
            # Always save non-DDP weights for UNet
            unet_state = unet.module.state_dict() if use_ddp else unet.state_dict()
            torch.save({
                'epoch': epoch + 1,
                'unet_state_dict': unet_state,
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(checkpoint_dir, f"ldm_epoch_{epoch+1}.pt"))
            print(f"Checkpoint saved: ldm_epoch_{epoch+1}.pt")
    cv2.destroyAllWindows()
    print("Latent diffusion training complete.")
finally:
    # Clean up distributed training
    cleanup_distributed()