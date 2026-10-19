import pyvista as pv

mesh = pv.read("ItemLowPoly.obj")
mesh = mesh.clean().compute_normals(auto_orient_normals=True)

plotter = pv.Plotter(window_size=(1000, 700))
plotter.set_background("white")

plotter.add_mesh(
    mesh,
    pbr=True,
    metallic=0.95,
    roughness=0.25,   # 0.15 shinier, 0.35 more brushed
    color="white",
    smooth_shading=True,
)

# Optional lighting (helps metal look realistic)
plotter.add_light(pv.Light(position=(2, 1, 3), light_type='scene light', intensity=1.0))
plotter.add_light(pv.Light(position=(-3, -2, 1), light_type='scene light', intensity=0.6))
# If available: env = pv.read_texture("studio_env.hdr"); plotter.set_environment_texture(env, True)

plotter.show()