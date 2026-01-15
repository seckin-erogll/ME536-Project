import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import tkinter as tk
from tkinter import ttk
import torch
import torch.nn as nn

IMG_SIZE = 64
LATENT_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Same model definition (must match training) ----
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

def pil_to_tk(img_pil):
    # Tk wants PhotoImage via PIL.ImageTk, import lazily
    from PIL import ImageTk
    return ImageTk.PhotoImage(img_pil)

def preprocess_drawing(img_rgba):
    """
    Image processing step:
    - convert to grayscale
    - invert (user draws black on white; keep as dark strokes)
    - slight blur + threshold-ish
    - dilation to thicken strokes
    - convert back to RGB on white bg
    """
    g = img_rgba.convert("L")
    g = ImageOps.invert(g)
    g = g.filter(ImageFilter.GaussianBlur(radius=0.6))
    # pseudo-threshold
    g = g.point(lambda p: 255 if p > 30 else 0)
    # thicken strokes
    g = g.filter(ImageFilter.MaxFilter(size=3))
    # invert back to black strokes on white
    g = ImageOps.invert(g)
    rgb = Image.merge("RGB", (g,g,g))
    return rgb

class App:
    def __init__(self, root):
        self.root = root
        root.title("Draw Emoji -> Match Trained Emoji (Autoencoder latent NN)")

        # Load model + latent DB
        self.model = ConvAE(LATENT_DIM).to(DEVICE)
        self.model.load_state_dict(torch.load("artifacts/emoji_ae.pt", map_location=DEVICE))
        self.model.eval()

        self.Z = np.load("artifacts/latent_Z.npy")  # (N, d)
        with open("artifacts/names.txt", "r", encoding="utf-8") as f:
            self.names = [line.strip() for line in f.readlines()]

        self.data_dir = "openmoji-72x72-black"

        # Canvas draw area
        self.canvas_size = 256
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.brush = 10
        self.last = None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "last", None))

        # Offscreen image to capture drawing
        self.img = Image.new("RGBA", (self.canvas_size, self.canvas_size), (255,255,255,255))
        self.draw = ImageDraw.Draw(self.img)

        # Right panel: results
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="n", padx=10, pady=10)

        self.lbl_match = ttk.Label(right, text="Nearest emoji: -")
        self.lbl_match.pack(pady=6)

        self.out1 = ttk.Label(right)
        self.out1.pack(pady=6)
        self.out2 = ttk.Label(right)
        self.out2.pack(pady=6)

        btns = ttk.Frame(right)
        btns.pack(pady=8, fill="x")

        ttk.Button(btns, text="Predict / Match", command=self.predict).pack(side="left", padx=4)
        ttk.Button(btns, text="Clear", command=self.clear).pack(side="left", padx=4)

        # show initial blank previews
        self.show_previews(Image.new("RGB",(IMG_SIZE,IMG_SIZE),(255,255,255)),
                           Image.new("RGB",(IMG_SIZE,IMG_SIZE),(255,255,255)))

    def paint(self, e):
        if self.last is None:
            self.last = (e.x, e.y)
            return
        x0, y0 = self.last
        x1, y1 = e.x, e.y
        self.canvas.create_line(x0, y0, x1, y1, width=self.brush, fill="black", capstyle=tk.ROUND, smooth=True)
        self.draw.line([x0, y0, x1, y1], fill=(0,0,0,255), width=self.brush)
        self.last = (x1, y1)

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("RGBA", (self.canvas_size, self.canvas_size), (255,255,255,255))
        self.draw = ImageDraw.Draw(self.img)
        self.lbl_match.config(text="Nearest emoji: -")

    def show_previews(self, user64_rgb, recon64_rgb):
        # enlarge for UI
        u = user64_rgb.resize((128,128), Image.NEAREST)
        r = recon64_rgb.resize((128,128), Image.NEAREST)
        self.tk_u = pil_to_tk(u)
        self.tk_r = pil_to_tk(r)
        self.out1.configure(image=self.tk_u, text="")  # keep refs
        self.out2.configure(image=self.tk_r, text="")

    def predict(self):
        # downsample drawing -> 64x64
        user = self.img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        user_pp = preprocess_drawing(user)

        x = np.asarray(user_pp).astype(np.float32)/255.0
        x = np.transpose(x, (2,0,1))
        xt = torch.tensor(x).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            z = self.model.encode(xt).cpu().numpy()[0]  # (d,)

        # nearest neighbor in latent DB
        dists = np.sum((self.Z - z[None,:])**2, axis=1)
        idx = int(np.argmin(dists))
        name = self.names[idx]

        # Option A: show the actual nearest training emoji image
        p = os.path.join(self.data_dir, name)
        nearest = Image.open(p).convert("RGBA").resize((IMG_SIZE,IMG_SIZE), Image.LANCZOS)
        bg = Image.new("RGBA", nearest.size, (255,255,255,255))
        nearest = Image.alpha_composite(bg, nearest).convert("RGB")

        # Option B: decode the nearest latent (forces it onto manifold)
        zt = torch.tensor(self.Z[idx]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            xhat = self.model.decode(zt).cpu().numpy()[0]
        xhat = np.transpose(xhat, (1,2,0))
        recon = Image.fromarray((xhat*255).clip(0,255).astype(np.uint8))

        self.lbl_match.config(text=f"Nearest emoji: {name}")
        # left preview: your processed drawing; right preview: decoded reconstruction
        self.show_previews(user_pp.convert("RGB"), recon)

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()

