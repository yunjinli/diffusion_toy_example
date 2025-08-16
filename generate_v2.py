import os
import torch
import torchvision
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# ---------- Config ----------
CHECKPOINT_PATH   = os.environ.get("CHECKPOINT_PATH", "./checkpoints/ldm_epoch_best.pt")
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_SHAPE      = (4, 16, 16)   # 128x128 with sd-vae-ft-ema
VAE_SCALE         = 0.18215
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", 30))
GUIDANCE_SCALE    = float(os.environ.get("GUIDANCE_SCALE", 1.5))
PREDICTION_TYPE   = os.environ.get("PREDICTION_TYPE", "v_prediction")  # match training ("v_prediction" or "epsilon")

# ---------- Load models ----------
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)

unet = UNet2DConditionModel(
    sample_size=16,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    down_block_types=("CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"),
    up_block_types=("UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"),
    cross_attention_dim=text_encoder.config.hidden_size,
).to(DEVICE)

# Load trained weights (prefer EMA if present; otherwise raw)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
# state = ckpt.get("unet_ema_state_dict", None)
# if isinstance(state, dict) and "shadow" in state:
#     # state saved via EMA.state_dict(); actual tensors under 'shadow'
#     unet.load_state_dict(state["shadow"], strict=True)
# else:
if "unet_ema_state_dict" in ckpt:
    print("Loading EMA weights")
    ema_state = ckpt["unet_ema_state_dict"]

    # If you saved with EMA.state_dict(), weights live in ema_state["shadow"]
    if "shadow" in ema_state:
        shadow_weights = ema_state["shadow"]
        # unwrap to plain dict of parameter tensors
        shadow_weights = {k: v.to(DEVICE) for k, v in shadow_weights.items()}
        unet.load_state_dict(shadow_weights, strict=True)
    else:
        # if you stored ema.copy_to(model).state_dict(), you can just load directly
        unet.load_state_dict(ema_state, strict=True)
else:
    print("Loading raw UNet weights")
    unet.load_state_dict(ckpt["unet_state_dict"], strict=True)

vae.eval(); text_encoder.eval(); unet.eval()

# DPMSolver, prediction type must MATCH training
# scheduler = DDPMScheduler(
#                             num_train_timesteps=1000,
#                             beta_schedule="squaredcos_cap_v2",
#                             prediction_type="v_prediction",
#                         )
scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    prediction_type=PREDICTION_TYPE,
    algorithm_type="dpmsolver++",
)

@torch.no_grad()
def encode_prompts(prompts, negative_prompts=None, device=DEVICE):
    """Returns (cond_embeds, uncond_embeds) with shape [B, 77, hidden]."""
    if isinstance(prompts, str):
        prompts = [prompts]
    bsz = len(prompts)

    # conditional
    inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    cond = text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).last_hidden_state

    # unconditional
    negatives = [""] * bsz if negative_prompts is None else (
        negative_prompts if isinstance(negative_prompts, list) else [negative_prompts] * bsz
    )
    uncond_inputs = tokenizer(negatives, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    uncond = text_encoder(input_ids=uncond_inputs.input_ids, attention_mask=uncond_inputs.attention_mask).last_hidden_state
    return cond, uncond

def cfg_rescale(pred_cond, pred_uncond, scale, rescale=0.7):
    """
    SDXL-style CFG rescale to reduce over-saturation / mush.
    """
    guided = pred_uncond + scale * (pred_cond - pred_uncond)
    std_c = pred_cond.std(dim=list(range(1, pred_cond.ndim)), keepdim=True)
    std_g = guided.std(dim=list(range(1, guided.ndim)), keepdim=True) + 1e-6
    return rescale * guided * (std_c / std_g) + (1.0 - rescale) * guided

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
    Saves a grid to out_path and returns tensor in [0,1], NCHW.
    """
    cond, uncond = encode_prompts(prompts, negative_prompts, device)
    if isinstance(prompts, str):
        prompts = [prompts]
    bsz = len(prompts)

    scheduler.set_timesteps(num_inference_steps, device=device)

    # Random latents
    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    latents = torch.randn((bsz, *LATENT_SHAPE), generator=generator, device=device)

    # Denoising loop with CFG (+ rescale)
    for t in scheduler.timesteps:
        lat_in = torch.cat([latents, latents], dim=0)
        context = torch.cat([uncond, cond], dim=0)
        model_pred = unet(lat_in, t, encoder_hidden_states=context).sample  # predicts v or eps
        pred_uncond, pred_cond = model_pred.chunk(2, dim=0)

        if guidance_scale is not None and guidance_scale != 1.0:
            guided = cfg_rescale(pred_cond, pred_uncond, guidance_scale, rescale=0.7)
        else:
            guided = pred_cond  # effectively CFG=1

        out = scheduler.step(guided, t, latents)
        latents = out.prev_sample

    # Decode with VAE
    imgs = vae.decode(latents / VAE_SCALE).sample
    imgs = imgs.clamp(-1, 1)
    imgs = (imgs + 1) / 2  # -> [0,1], NCHW

    torchvision.utils.save_image(imgs, out_path, nrow=min(bsz, 8))
    print(f"Saved: {out_path}")
    return imgs

if __name__ == "__main__":
    samples = generate(
        prompts=[
            "young woman, smiling",
            "old man, gray hair, glasses",
            "man, black hair, mustache",
        ],
        negative_prompts="blurry, low quality",
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        seed=42,
        out_path="preview_grid.png",
    )