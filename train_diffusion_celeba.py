import os
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from prepare_celeba import CelebATextDataset, IMG_DIR, ATTR_PATH
from torchinfo import summary
import numpy as np  
# Hyperparameters
BATCH_SIZE = 8
IMG_SIZE = 128
EPOCHS = 2
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load text encoder and tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE)
# Load UNet and scheduler
unet = UNet2DConditionModel(
    sample_size=IMG_SIZE,
    in_channels=3,
    out_channels=3,
    layers_per_block=1,
    block_out_channels=(64, 128, 256),
    down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"
    ),
    cross_attention_dim=text_encoder.config.hidden_size,
).to(DEVICE)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

print(summary(unet))

# Load CelebA dataset
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
dataset = CelebATextDataset(IMG_DIR, ATTR_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer
optimizer = torch.optim.Adam(list(unet.parameters()) + list(text_encoder.parameters()), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for i, (captions, images) in enumerate(pbar):
        images = images.to(DEVICE)
        # Tokenize and encode text
        inputs = tokenizer(list(captions), padding="max_length", max_length=77, return_tensors="pt")
        input_ids = inputs.input_ids.to(DEVICE)
        attention_mask = inputs.attention_mask.to(DEVICE)
        text_embeds = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Sample random timesteps
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=DEVICE)
        noise = torch.randn_like(images)
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

        # Predict noise
        noise_pred = unet(noisy_images, timesteps, text_embeds).sample

        # Loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix({"loss": loss.item()})

        # Show denoised image panel every 5 steps
        if i % 5 == 0:
            with torch.no_grad():
                # Denoised images (simple subtraction)
                denoised_batch = (noisy_images - noise_pred).clamp(0, 1).cpu()
                gt_batch = images.cpu()
                panels = []
                for idx in range(denoised_batch.shape[0]):
                    denoised = denoised_batch[idx].permute(1, 2, 0).numpy()
                    gt = gt_batch[idx].permute(1, 2, 0).numpy()
                    denoised_bgr = cv2.cvtColor((denoised * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
                    gt_bgr = cv2.cvtColor((gt * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
                    panels.append(cv2.hconcat([denoised_bgr, gt_bgr]))
                panel = cv2.vconcat(panels)
                cv2.imshow('Denoised | Ground Truth', panel)
                cv2.waitKey(1)

    print(f"Epoch {epoch+1} finished.")

cv2.destroyAllWindows()
print("Training complete.")
