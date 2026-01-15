import os, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

IMG_SIZE = 64
LATENT_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "openmoji-72x72-black"

class ConvAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h)

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        return self.dec(h)

def load_img(p):
    img = Image.open(p).convert("RGBA").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    bg = Image.new("RGBA", img.size, (255,255,255,255))
    img = Image.alpha_composite(bg, img).convert("RGB")
    x = np.asarray(img).astype(np.float32) / 255.0
    x = np.transpose(x, (2,0,1))
    return torch.tensor(x).unsqueeze(0)

def save_tensor_img(x, outpath):
    x = (x.squeeze(0).permute(1,2,0).cpu().numpy() * 255.0).clip(0,255).astype(np.uint8)
    Image.fromarray(x).save(outpath)

def main():
    model = ConvAE(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load("artifacts/emoji_ae.pt", map_location=DEVICE))
    model.eval()

    paths = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.png"), recursive=True))[:8]
    os.makedirs("artifacts/recon", exist_ok=True)

    with torch.no_grad():
        for i, p in enumerate(paths):
            x = load_img(p).to(DEVICE)
            z = model.encode(x)
            xhat = model.decode(z)
            save_tensor_img(x, f"artifacts/recon/{i:02d}_orig.png")
            save_tensor_img(xhat, f"artifacts/recon/{i:02d}_recon.png")

    print("Wrote artifacts/recon/*_orig.png and *_recon.png")

if __name__ == "__main__":
    main()

