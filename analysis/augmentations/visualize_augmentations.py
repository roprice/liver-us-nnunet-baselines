"""
Visualize nnU-Net's default augmentations applied to a liver ultrasound image.

Shows 10 individual transforms side by side with the original,
using the exact same parameters nnU-Net applies during training.

Usage:
    pip install numpy Pillow scipy scikit-image
    python visualize_augmentations.py --image path/to/image.jpg

Outputs a grid image to augmentation_examples.png
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.transform import resize


def load_and_prep(path):
    """Load image as grayscale float array normalized to 0-1."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def to_pil(arr):
    """Convert 0-1 float array back to PIL Image."""
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

# --- Individual augmentations matching nnU-Net defaults ---

def aug_rotation(img, angle_deg=47):
    """Rotation. nnU-Net: p=0.2, ±180° for 2D."""
    pil = to_pil(img)
    rotated = pil.rotate(angle_deg, resample=Image.BILINEAR, fillcolor=0)
    return np.array(rotated, dtype=np.float32) / 255.0


def aug_scaling(img, scale=0.75):
    """Scaling. nnU-Net: p=0.2, range 0.7x to 1.4x."""
    h, w = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    resized = resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True)
    # Center in original canvas
    out = np.zeros_like(img)
    y_off = (h - new_h) // 2
    x_off = (w - new_w) // 2
    out[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return out


def aug_gaussian_noise(img, variance=0.05):
    """Gaussian noise. nnU-Net: p=0.1, variance 0 to 0.1."""
    noise = np.random.normal(0, np.sqrt(variance), img.shape).astype(np.float32)
    return img + noise


def aug_gaussian_blur(img, sigma=0.8):
    """Gaussian blur. nnU-Net: p=0.2, sigma 0.5 to 1.0."""
    return gaussian_filter(img, sigma=sigma)


def aug_brightness(img, multiplier=1.2):
    """Multiplicative brightness. nnU-Net: p=0.15, range 0.75 to 1.25."""
    return img * multiplier


def aug_contrast(img, factor=0.8):
    """Contrast adjustment. nnU-Net: p=0.15, range 0.75 to 1.25."""
    mean = img.mean()
    return (img - mean) * factor + mean


def aug_low_resolution(img, scale=0.5):
    """Low-resolution simulation. nnU-Net: p=0.25, downscale 0.5x to 1.0x."""
    h, w = img.shape
    small = resize(img, (int(h * scale), int(w * scale)),
                   anti_aliasing=True, preserve_range=True)
    return resize(small, (h, w), order=0, preserve_range=True)  # nearest neighbor upsample


def aug_gamma(img, gamma=1.4):
    """Gamma correction. nnU-Net: p=0.3, gamma 0.7 to 1.5."""
    return np.power(np.clip(img, 1e-8, 1.0), gamma)


def aug_gamma_inverted(img, gamma=1.3):
    """Inverted gamma correction. nnU-Net: p=0.1, gamma 0.7 to 1.5.
    Inverts intensity, applies gamma, inverts back."""
    inverted = 1.0 - img
    corrected = np.power(np.clip(inverted, 1e-8, 1.0), gamma)
    return 1.0 - corrected


def aug_mirror_horizontal(img):
    """Horizontal flip. nnU-Net: always enabled, axis 1."""
    return np.fliplr(img)


def aug_mirror_vertical(img):
    """Vertical flip. nnU-Net: always enabled, axis 0."""
    return np.flipud(img)


def aug_combined(img):
    """A plausible combination: rotation + brightness + gamma + horizontal flip.
    This is the kind of thing nnU-Net might produce on any given training step."""
    out = aug_rotation(img, angle_deg=-23)
    out = aug_brightness(out, multiplier=1.15)
    out = aug_gamma(out, gamma=0.8)
    out = aug_mirror_horizontal(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to ultrasound image")
    parser.add_argument("--output", default="augmentation_examples.png")
    args = parser.parse_args()

    img = load_and_prep(args.image)

    augmentations = [
        ("Original", img),
        ("Rotation (47°)", aug_rotation(img, 47)),
        ("Scale (0.75x)", aug_scaling(img, 0.75)),
        ("Gaussian noise\n(var=0.05)", aug_gaussian_noise(img, 0.05)),
        ("Gaussian blur\n(σ=0.8)", aug_gaussian_blur(img, 0.8)),
        ("Brightness\n(1.2x)", aug_brightness(img, 1.2)),
        ("Contrast\n(0.8x)", aug_contrast(img, 0.8)),
        ("Low-res sim\n(0.5x)", aug_low_resolution(img, 0.5)),
        ("Gamma (1.4)", aug_gamma(img, 1.4)),
        ("Gamma inverted\n(1.3)", aug_gamma_inverted(img, 1.3)),
        ("Mirror (horiz)", aug_mirror_horizontal(img)),
        ("Mirror (vert)", aug_mirror_vertical(img)),
        ("Combined\n(rotation+bright\n+gamma+flip)", aug_combined(img)),
    ]

    # Layout: arrange in rows
    n = len(augmentations)
    cols = 4
    rows = (n + cols - 1) // cols

    # Thumbnail size
    thumb_h, thumb_w = 300, 380
    pad = 10
    label_h = 60

    canvas_w = cols * (thumb_w + pad) + pad
    canvas_h = rows * (thumb_h + label_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    for idx, (name, aug_img) in enumerate(augmentations):
        row = idx // cols
        col = idx % cols

        x = pad + col * (thumb_w + pad)
        y = pad + row * (thumb_h + label_h + pad)

        # Resize augmented image to thumbnail
        pil_img = to_pil(aug_img)
        pil_img = pil_img.resize((thumb_w, thumb_h), Image.BILINEAR)

        # Green border for original
        if idx == 0:
            border = Image.new("RGB", (thumb_w + 4, thumb_h + 4), (80, 220, 80))
            border.paste(pil_img.convert("RGB"), (2, 2))
            canvas.paste(border, (x - 2, y - 2))
        else:
            canvas.paste(pil_img.convert("RGB"), (x, y))

        # Label
        for i, line in enumerate(name.split("\n")):
            draw.text((x + 4, y + thumb_h + 4 + i * 16), line, fill=(200, 200, 200))

    canvas.save(args.output, quality=95)
    print(f"Saved to {args.output}")
    print(f"Grid: {cols} cols x {rows} rows, {n} augmentations")


if __name__ == "__main__":
    main()
