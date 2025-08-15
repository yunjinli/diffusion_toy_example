import os
import torch
import torchvision
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# ---------- Config ----------
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "./checkpoints/ldm_epoch_best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_SHAPE = (4, 16, 16)  # (C, H, W) for 128x128 images with sd-vae-ft-ema
VAE_SCALE = 0.18215
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", 50))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", 7.5))
SCHEDULER_NAME = os.environ.get("SCHEDULER", "ddpm").lower()

# ---------- Load models ----------
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)

unet = UNet2DConditionModel(
    sample_size=16,
    in_channels=4,
    out_channels=4,
    layers_per_block=3,
    block_out_channels=(128, 256, 512, 512),
    down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
    cross_attention_dim=text_encoder.config.hidden_size,
).to(DEVICE)

# Load your trained UNet weights
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
unet.load_state_dict(ckpt["unet_state_dict"])
vae.eval(); text_encoder.eval(); unet.eval()

# Diffusion scheduler (DDPM by default, DDIM optional)
if SCHEDULER_NAME == "ddim":
    scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
else:
    scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")

@torch.no_grad()
def encode_prompts(prompts, negative_prompts=None, device=DEVICE):
    """
    Returns (cond_embeds, uncond_embeds) with shape [B, 77, hidden].
    If negative_prompts is None, uses empty strings for unconditional branch.
    """
    if isinstance(prompts, str): prompts = [prompts]
    bsz = len(prompts)

    # conditional
    inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    cond = text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).last_hidden_state

    # unconditional
    if negative_prompts is None:
        negatives = [""] * bsz
    else:
        negatives = negative_prompts if isinstance(negative_prompts, list) else [negative_prompts] * bsz

    uncond_inputs = tokenizer(negatives, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    uncond = text_encoder(input_ids=uncond_inputs.input_ids, attention_mask=uncond_inputs.attention_mask).last_hidden_state
    return cond, uncond

@torch.no_grad()
def generate(
    prompts,
    negative_prompts=None,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    seed=None,
    out_path="samples.png",
    device=DEVICE,
):
    """
    prompts: str or List[str]
    Saves a grid to out_path and returns the tensor in [0,1].
    """
    cond, uncond = encode_prompts(prompts, negative_prompts, device)
    if isinstance(prompts, str): prompts = [prompts]
    bsz = len(prompts)

    # Prepare scheduler timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Random latents
    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    latents = torch.randn((bsz, *LATENT_SHAPE), generator=generator, device=device)

    # Denoising loop with CFG
    for t in scheduler.timesteps:
        # Duplicate latents for uncond/cond
        lat_in = torch.cat([latents, latents], dim=0)
        context = torch.cat([uncond, cond], dim=0)

        # UNet predicts ε
        noise_pred = unet(lat_in, t, encoder_hidden_states=context).sample
        noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode with VAE
    imgs = vae.decode(latents / VAE_SCALE).sample
    imgs = imgs.clamp(-1, 1)
    imgs = (imgs + 1) / 2  # -> [0,1], NCHW

    # Save a grid
    torchvision.utils.save_image(imgs, out_path, nrow=min(bsz, 8))
    print(f"Saved: {out_path}")
    return imgs

if __name__ == "__main__":
    # Examples
    samples = generate(
        prompts=[
            "A portrait of a young woman smiling with blond hair",
            "A portrait of a male wearing glasses with a five o'clock shadow",
            "A portrait of a person with wavy hair, rosy cheeks, and wearing a hat",
        ],
        negative_prompts="blurry, low quality",
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        seed=42,
        out_path="preview_grid.png",
    )
