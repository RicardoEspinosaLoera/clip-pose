import numpy as np
from PIL import Image

# Albedo (flat gray, base color of steel)
albedo = np.full((512, 512, 3), 220, dtype=np.uint8)  # light gray steel
Image.fromarray(albedo).save("albedo.png")

# Normal map (flat normal = blue)
normal = np.zeros((512, 512, 3), dtype=np.uint8)
normal[..., 0] = 128
normal[..., 1] = 128
normal[..., 2] = 255
Image.fromarray(normal).save("normal.png")

# Roughness/Metallic packed (R=roughness, G=metallic, B unused)
rmetal = np.zeros((512, 512, 3), dtype=np.uint8)
rmetal[..., 0] = 64   # low roughness (shinier)
rmetal[..., 1] = 255  # fully metallic
rmetal[..., 2] = 0
Image.fromarray(rmetal).save("rmetal.png")