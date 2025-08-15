import os
import requests
import zipfile
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

CELEBA_IMG_URL = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
CELEBA_ATTR_URL = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pblRyaVFSWGxPY0U"
DATA_DIR = os.environ.get('DATA_DIR', "/mnt/sda/celeba")
IMG_DIR = os.path.join(DATA_DIR, "img_align_celeba")
ATTR_PATH = os.path.join(DATA_DIR, "list_attr_celeba.txt")

# Download helper
def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {dest}...")
        response = requests.get(url, stream=True)
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {dest}")
    else:
        print(f"{dest} already exists.")

# Unzip helper
def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

# Download images and attributes
os.makedirs(DATA_DIR, exist_ok=True)

# Images (manual download recommended due to Google Drive restrictions)
print(f"Please manually download 'img_align_celeba.zip' from the official CelebA site and place it in {DATA_DIR}.")
if os.path.exists(os.path.join(DATA_DIR, "img_align_celeba.zip")) and not os.path.exists(IMG_DIR):
    unzip_file(os.path.join(DATA_DIR, "img_align_celeba.zip"), DATA_DIR)

# Attributes
if not os.path.exists(ATTR_PATH):
    print(f"Please manually download 'list_attr_celeba.txt' from the official CelebA site and place it in {DATA_DIR}.")

# Attribute to caption
ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

# def attr_to_caption(attr_row):
#     attrs = [ATTRIBUTES[i] for i, v in enumerate(attr_row) if v == 1]
#     if not attrs:
#         return "A face."
#     return "A face with " + ", ".join(attrs) + "."

ATTR_READABLE = {
    "5_o_Clock_Shadow": "a five o'clock shadow",
    "Arched_Eyebrows": "arched eyebrows",
    "Attractive": "attractive",
    "Bags_Under_Eyes": "bags under the eyes",
    "Bald": "bald",
    "Bangs": "bangs",
    "Big_Lips": "big lips",
    "Big_Nose": "a big nose",
    "Black_Hair": "black hair",
    "Blond_Hair": "blond hair",
    "Blurry": "blurry",
    "Brown_Hair": "brown hair",
    "Bushy_Eyebrows": "bushy eyebrows",
    "Chubby": "chubby cheeks",
    "Double_Chin": "a double chin",
    "Eyeglasses": "wearing glasses",
    "Goatee": "a goatee",
    "Gray_Hair": "gray hair",
    "Heavy_Makeup": "wearing heavy makeup",
    "High_Cheekbones": "high cheekbones",
    "Male": "male",
    "Mouth_Slightly_Open": "mouth slightly open",
    "Mustache": "a mustache",
    "Narrow_Eyes": "narrow eyes",
    "No_Beard": "no beard",
    "Oval_Face": "an oval face",
    "Pale_Skin": "pale skin",
    "Pointy_Nose": "a pointy nose",
    "Receding_Hairline": "a receding hairline",
    "Rosy_Cheeks": "rosy cheeks",
    "Sideburns": "sideburns",
    "Smiling": "smiling",
    "Straight_Hair": "straight hair",
    "Wavy_Hair": "wavy hair",
    "Wearing_Earrings": "wearing earrings",
    "Wearing_Hat": "wearing a hat",
    "Wearing_Lipstick": "wearing lipstick",
    "Wearing_Necklace": "wearing a necklace",
    "Wearing_Necktie": "wearing a necktie",
    "Young": "young"
}

def attr_to_caption(attr_row):
    attrs = [ATTR_READABLE[ATTRIBUTES[i]] for i, v in enumerate(attr_row) if v == 1]
    if not attrs:
        return "A portrait of a person."
    return "A portrait of a person with " + ", ".join(attrs) + "."


# PyTorch Dataset
class CelebATextDataset(Dataset):
    def __init__(self, img_dir, attr_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # Load attributes
        df = pd.read_csv(attr_path, sep=r"\s+", skiprows=1)
        df = df.reset_index().rename(columns={"index": "image_id"})
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image = Image.open(img_path).convert("RGB")
        attr_row = row[1:].values.astype(int)
        caption = attr_to_caption(attr_row)
        if self.transform:
            image = self.transform(image)
        return caption, image

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = CelebATextDataset(IMG_DIR, ATTR_PATH, transform=transform)
    print("Example caption/image:")
    caption, image = dataset[0]
    print(caption)
    print(image.shape)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.show()