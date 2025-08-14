import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import cv2
import numpy as np

# Paths to checkpoints
CHECKPOINT_PATH = "checkpoints/ldm_epoch_1.pt"  # Change to your desired checkpoint

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
unet = UNet2DConditionModel(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    layers_per_block=1,
    block_out_channels=(128, 256, 512),
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

# Load trained weights
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
unet.load_state_dict(ckpt['unet_state_dict'])
# text_encoder.load_state_dict(ckpt['text_encoder_state_dict'])

# Sampling function
def sample_ldm(prompt, num_steps=50, latent_shape=(1, 4, 32, 32)):
    # Encode prompt
    inputs = tokenizer([prompt], padding="max_length", truncation=True, max_length=77, return_tensors="pt")
    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)
    text_embeds = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # Start from pure noise in latent space
    latents = torch.randn(latent_shape, device=DEVICE)
    for t in reversed(range(num_steps)):
        timesteps = torch.full((latents.shape[0],), t, device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            noise_pred = unet(latents, timesteps, text_embeds).sample
        latents = noise_scheduler.step(noise_pred, timesteps, latents).prev_sample
    # Decode to image
    with torch.no_grad():
        imgs = vae.decode(latents / 0.18215).sample.clamp(0, 1).cpu()
    # Convert to displayable format
    img = imgs[0].permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow('Generated Image', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

if __name__ == "__main__":
    prompt = input("Enter a text prompt: ")
    sample_ldm(prompt)
