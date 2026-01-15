import  glob, random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import os  # <-- missing

# ---------- Paths / Image ----------
DATA_DIR = "openmoji-72x72-black"   # or "data/emojis50" if you have that folder
IMG_SIZE = 64                       # your model assumes 64x64

# ---------- Config ----------
BATCH = 16
EPOCHS = 200
LR = 1e-3
LATENT_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- Dataset ----------
class EmojiDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted(glob.glob(os.path.join(folder, "*.png")))
        if len(self.paths) < 10:
            raise RuntimeError(f"Not enough PNGs in {folder}. Found {len(self.paths)}.")
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGBA")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        # white background (important)
        bg = Image.new("RGBA", img.size, (255,255,255,255))
        img = Image.alpha_composite(bg, img).convert("RGB")
        x = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
        x = np.transpose(x, (2,0,1))                   # (3,H,W)
        return torch.tensor(x), os.path.basename(p)

# ---------- Model ----------
class ConvAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder: 3x64x64 -> latent
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),      # 32x32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),     # 64x16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),    # 128x8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),   # 256x4x4
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)

        # Decoder: latent -> 3x64x64
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),  # 128x8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),   # 64x16x16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),    # 32x32x32
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),  # 3x64x64
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        z = self.fc_mu(h)
        return z

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        xhat = self.dec(h)
        return xhat

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z

def main():
    ds = EmojiDataset(DATA_DIR)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

    model = ConvAE(LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(1, EPOCHS+1):
        losses = []
        for x, _ in dl:
            x = x.to(DEVICE)
            xhat, _ = model(x)
            loss = loss_fn(xhat, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if ep % 20 == 0 or ep == 1:
            print(f"epoch {ep:04d} | loss {np.mean(losses):.6f}")

    # ---- Build latent database for nearest-neighbor matching ----
    model.eval()
    all_z, all_names = [], []
    with torch.no_grad():
        for x, names in DataLoader(ds, batch_size=32, shuffle=False):
            z = model.encode(x.to(DEVICE)).cpu().numpy()
            all_z.append(z)
            all_names.extend(list(names))
    Z = np.concatenate(all_z, axis=0)  # (N, latent_dim)

    # PCA (for visualization / feature analysis)
    pca = PCA(n_components=2, random_state=0).fit(Z)
    Z2 = pca.transform(Z)

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/emoji_ae.pt")
    np.save("artifacts/latent_Z.npy", Z)
    np.save("artifacts/latent_Z2_pca.npy", Z2)
    with open("artifacts/names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_names))

    print("Saved: artifacts/emoji_ae.pt, latent_Z.npy, latent_Z2_pca.npy, names.txt")

if __name__ == "__main__":
    main()

