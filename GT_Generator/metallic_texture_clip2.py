import numpy as np
from PIL import Image
import pyvista as pv

# --- 1) Load mesh
mesh = pv.read("ItemLowPoly.obj").clean().compute_normals(auto_orient_normals=True)

# --- 2) Load textures you already have
albedo_path    = "Metal/Metal049A_1K-JPG_Color.jpg"
normal_path    = "Metal/Metal049A_1K-JPG_NormalGL.jpg"   # Use the *GL* normal
roughness_path = "Metal/Metal049A_1K-JPG_Roughness.jpg"
metallic_path  = "Metal/Metal049A_1K-JPG_Metalness.jpg"

# Read with PIL to pack ORM
albedo_img    = Image.open(albedo_path).convert("RGB")
normal_img    = Image.open(normal_path).convert("RGB")
roughness_img = Image.open(roughness_path).convert("L")   # single channel
metallic_img  = Image.open(metallic_path).convert("L")

# Ensure same size (resize rough/metal to albedo size if needed)
W, H = albedo_img.size
if roughness_img.size != (W, H):
    roughness_img = roughness_img.resize((W, H), Image.BILINEAR)
if metallic_img.size != (W, H):
    metallic_img = metallic_img.resize((W, H), Image.BILINEAR)
if normal_img.size != (W, H):
    normal_img = normal_img.resize((W, H), Image.BILINEAR)

# --- 3) Pack ORM: R=Occlusion(255), G=Roughness, B=Metallic
occlusion = Image.new("L", (W, H), color=255)  # if you don't have AO
orm_img = Image.merge("RGB", (occlusion, roughness_img, metallic_img))

# Convert PIL -> numpy (uint8) -> PyVista vtkTexture
albedo_np = np.array(albedo_img, dtype=np.uint8)
normal_np = np.array(normal_img, dtype=np.uint8)
orm_np    = np.array(orm_img,    dtype=np.uint8)

albedo_tex = pv.numpy_to_texture(albedo_np)
normal_tex = pv.numpy_to_texture(normal_np)
orm_tex    = pv.numpy_to_texture(orm_np)

# --- 4) Plot and attach via VTK property
plotter = pv.Plotter(window_size=(1000, 700))
plotter.set_background("white")

actor = plotter.add_mesh(
    mesh,
    pbr=True,              # enables PBR shading on vtkProperty
    texture=albedo_tex,    # base color
    smooth_shading=True,
)

# Access VTK property and set the extra PBR textures
prop = actor.GetProperty() if hasattr(actor, "GetProperty") else actor.prop
prop.SetBaseColorTexture(albedo_tex)
prop.SetNormalTexture(normal_tex)
prop.SetORMTexture(orm_tex)          # (R=AO, G=Roughness, B=Metallic)

# Optional: lights / environment for better metal reflections
#plotter.add_light(pv.Light(position=(2, 1, 3), light_type='scene light', intensity=1.0))
#plotter.add_light(pv.Light(position=(-3, -2, 1), light_type='scene light', intensity=0.6))
headlight = pv.Light(light_type='headlight', intensity=1.2, color='white')
plotter.add_light(headlight)
# If you have an HDRI:
# env = pv.read_texture("studio_env.hdr"); plotter.set_environment_texture(env, True)

plotter.show()
