import pyvista as pv
import numpy as np
from vtk import vtkMatrix4x4, vtkMatrix3x3
import cv2
from scipy.spatial.transform import Rotation as scipyR
np.set_printoptions(precision=2)
import os
from dataclasses import dataclass
import json
import random
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
from typing import Optional


#WORLD_UP  = np.array([1., 0., 0.])   # world up (+Z)
MODEL_UP  = np.array([1., 0., 0.])   # model's up (+Z)
MIN_UP_DOT = np.cos(np.deg2rad(30))  # reject if model-up tilts >30° from world-up
path = "D:\\Harvard\\implementations\\GT_Generator\\Data\\noclips"  # Replace with the desired directory path
BACKGROUND_IMAGES = os.listdir(path)

# ------------------- CONFIG -------------------
OUT_DIR = "D:\\Harvard\\implementations\\Aneurysm_clip_pose\\data\\test\\"
os.makedirs(OUT_DIR, exist_ok=True)

w = 800
h = 544
znear = 0.01
zfar = 1000.0

MESH_PATH = 'Item.obj'
#TEX_PATH  = 'Data/noclips/1.png'

WORLD_UP = np.array([0.0, 0.0, 1.0])  # define "up" = +Z
MAX_TILT_DEG = 15.0                   # how far from straight-down
MAX_ROLL_DEG = 10.0                   # in-plane roll limit

# ------------------- SCENE -------------------
plotter = pv.Plotter(off_screen=False, window_size=[w, h])
mesh = pv.read(MESH_PATH)

def bbox_corners_world(mesh):
    xmin,xmax,ymin,ymax,zmin,zmax = mesh.bounds
    return np.array([
        [xmin,ymin,zmin],[xmin,ymin,zmax],[xmin,ymax,zmin],[xmin,ymax,zmax],
        [xmax,ymin,zmin],[xmax,ymin,zmax],[xmax,ymax,zmin],[xmax,ymax,zmax]
    ], dtype=float)

CORNERS_W = bbox_corners_world(mesh)

# Lights
plotter.camera.clipping_range = (znear, zfar)
headlight = pv.Light(light_type='headlight', intensity=40.0, color='white')
plotter.add_light(headlight)

# Quit key
class PlotterControls:
    def __init__(self, plotter):
        self.p = plotter
        # cross-version: add_key_event vs add_key_event_handler
        if hasattr(self.p, "add_key_event_handler"):
            self.p.add_key_event_handler("q", self.quit)
        else:
            self.p.add_key_event("q", self.quit)
        self.stop = False
    def quit(self):
        print("Stopped")
        self.stop = True

pc = PlotterControls(plotter)

# ------------------- DATA (your 15 samples) -------------------
SAMPLES = [
    dict(tvec=[-9.31,  2.95, -72.77],
         rotMat=[[-0.21, 0.95, 0.22], [-0.23,-0.27, 0.93], [0.95, 0.15, 0.28]],
         camera_position=[(67.75670394105907, 20.214820394260954, 19.770608665197702),
                          (-0.40904115432158195, 9.79354414355711, -0.41251088126070284),
                          (-0.23474881600379136, -0.27193729120690946, 0.9332433246671884)]),
    dict(tvec=[ 3.64, -2.24, -44.53],
         rotMat=[[ 0.05,-0.99,-0.09], [-0.44, 0.06,-0.89], [0.89, 0.08,-0.44]],
         camera_position=[(38.67890932229465, 7.460361129380067, -21.209973306363004),
                          (2.392845682212369, 4.089967462217455, -3.408197414569912),
                          (-0.444314866055916, 0.06370962092015958, -0.8936024753792503)]),
    dict(tvec=[-7.48,  3.80, -68.72],
         rotMat=[[-0.24, 0.93, 0.27], [-0.22,-0.32, 0.92], [0.95, 0.16, 0.28]],
         camera_position=[(63.99561357182423,19.435715587249124,17.87528516370878),
                          (-0.09666159364077331,8.345702977110214,-1.244465558093614),
                          (-0.21940564614191074,-0.32061683667377483,0.9214477774037726)]),
    dict(tvec=[-7.21,  3.31, -67.39],
         rotMat=[[ 0.23, 0.41, 0.88], [ 0.08,-0.91, 0.40], [0.97,-0.02,-0.25]],
         camera_position=[(66.69903687450233, 4.532093269630731, -11.622317148278487),
                          (3.6594059811416333, 5.925828451909851, 4.446543541165942),
                          (0.08249080190518086, -0.9116126875328252, 0.40268793814841264)]),
    dict(tvec=[ 2.78, -0.60, -34.20],
         rotMat=[[-0.62,-0.74, 0.28], [-0.13,-0.25,-0.96], [0.78,-0.63, 0.05]],
         camera_position=[(28.195998077609474,-19.55837649039016,0.5021879300174465),
                          (2.159917127818674,1.4744169709542223,-1.3036514869370681),
                          (-0.13340927610869838,-0.247538545804042,-0.9596492241383687)]),
    dict(tvec=[ 10.76,-22.03,-132.64],
         rotMat=[[-0.22, 0.80, 0.56], [ 0.19,-0.52, 0.83], [0.95, 0.29,-0.04]],
         camera_position=[(133.35062020577126,18.985725353597324,7.254578949896062),
                          (8.063130794614748,-19.699239381154435,12.250966131273037),
                          (0.19448264643715946,-0.522655706061747,0.8300647644346166)]),
    dict(tvec=[ 10.76,-22.03,-132.64],
         rotMat=[[-0.22, 0.80, 0.56], [ 0.19,-0.52, 0.83], [0.95, 0.29,-0.04]],
         camera_position=[(133.35062020577126,18.985725353597324,7.254578949896062),
                          (8.063130794614748,-19.699239381154435,12.250966131273037),
                          (0.19448264643715946,-0.522655706061747,0.8300647644346166)]),
    dict(tvec=[ 0.66,-1.16,-34.64],
         rotMat=[[-0.24, 0.15, 0.96], [-0.12,-0.99, 0.12], [0.96,-0.08, 0.25]],
         camera_position=[(33.444953538558444,-4.145504194393513,8.132542986917276),
                          (3.199973417824173,-1.5206884989823073,0.32620721089742266),
                          (-0.11747353602473018,-0.985324895229425,0.12383383695488519)]),
    dict(tvec=[ -7.81,  7.71,-101.54],
         rotMat=[[-0.11, 0.27,-0.96], [-0.07, 0.96, 0.28], [0.99, 0.10,-0.08]],
         camera_position=[(100.39017126263931,4.458650179477533,-18.253737171664138),
                          (4.083571633612834,-4.8301451806900415,-10.006890713632709),
                          (-0.06829163715435918,0.9573382464338337,0.2807841416636092)]),
    dict(tvec=[ -9.60, -4.50, -49.64],
         rotMat=[[ 0.66, 0.74, 0.14], [-0.21, 0.00, 0.98], [0.73,-0.67, 0.16]],
         camera_position=[(41.40837936550699,-26.09210406351125,13.448008241910276),
                          (6.143114916637818,6.405999378206531,5.895175881083812),
                          (-0.20795355879851501,0.0016668560191337722,0.9781372802291324)]),
    dict(tvec=[ -0.33,  4.17, -39.69],
         rotMat=[[-0.38,-0.05, 0.92], [-0.21,-0.97,-0.14], [0.90,-0.25, 0.36]],
         camera_position=[(36.48771780867066,-5.815330645082126,15.090606443164976),
                          (1.5761925791458382,3.796366205343312,1.1932580304668179),
                          (-0.21228568479535068,-0.967698040157433,-0.13599739374867661)]),
    dict(tvec=[ -1.60,  6.60, -48.27],
         rotMat=[[-0.27, 0.06,-0.96], [-0.30, 0.94, 0.14], [0.92, 0.32,-0.24]],
         camera_position=[(45.73001924581199,9.505435228793942,-13.959649071020849),
                          (3.9750189444246185,-5.272678982966736,-3.1016240759788247),
                          (-0.29768974220551514,0.9442797335087737,0.140415819159429)]),
    dict(tvec=[ -13.01,  4.85, -83.36],
         rotMat=[[-0.12, 0.45, 0.88], [-0.13,-0.89, 0.44], [0.98,-0.06, 0.16]],
         camera_position=[(81.19413792477494,5.10622318635737,22.86129939792911),
                          (0.39416023952652934,10.11184104291692,9.600187804148026),
                          (-0.12695196622661023,-0.8902153873091296,0.43749258561632115)]),
    dict(tvec=[ -4.02,  2.51, -79.57],
         rotMat=[[-0.11,-0.99,-0.03], [0.22, 0.01,-0.98], [0.97,-0.12, 0.21]],
         camera_position=[(76.14959721954217,-13.46084175432511,19.35365013688427),
                          (3.975574153588986,-4.6216494215846735,3.4249746818242977),
                          (0.21626317239118548,0.006471150686077013,-0.9763136609082648)]),
    dict(tvec=[ -5.16,  1.46, -27.06],
         rotMat=[[-0.03,-1.00,-0.07], [-0.17, 0.07,-0.98], [0.99,-0.01,-0.17]],
         camera_position=[(26.792168597807986,-5.631560243164169,-3.4190827246214415),
                          (-3.9968562509338748,-5.1907899125833294,1.7904997189159693),
                          (-0.1654611851822502,0.06947469944825292,-0.9837661624261428)]),
]

def reconstruct_obj_world_from_json(cam, clip):
    """
    Reconstructs the world-space pose (R_obj_w, t_obj_w)
    from a GT JSON entry containing camera + clip.pose_se3.
    """
    # --- Load camera parameters ---
    pos = np.array(cam["position"], dtype=np.float32)
    focal = np.array(cam["focal_point"], dtype=np.float32)
    up = np.array(cam["view_up"], dtype=np.float32)

    # --- Build camera frame (same as PyVista) ---
    forward = focal - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # --- Camera-to-world rotation/translation ---
    R_c2w = np.stack([right, up, -forward], axis=1)   # VTK-style camera
    t_c2w = pos

    # --- Object-to-camera from JSON (R_co, t_co) ---
    q = clip["quaternion_wxyz"]
    R_co = Rscipy.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t_co = np.array(clip["translation_m"], dtype=np.float32).reshape(3, 1)

    # --- Back-project into world coordinates ---
    R_obj_w = R_c2w @ R_co
    t_obj_w = (R_c2w @ t_co).squeeze() + t_c2w

    return R_obj_w, t_obj_w

def intrinsics_from_vtk(camera, w, h):
    """
    Compute pixel intrinsics (fx, fy, cx, cy) from a PyVista/vtkCamera.
    Uses vertical field of view and accounts for aspect ratio.
    """
    fovy = np.deg2rad(camera.GetViewAngle())   # vertical FOV
    fy = (h / 2.0) / np.tan(fovy / 2.0)
    fx = fy * (w / h)                          # adjust for aspect ratio
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    return fx, fy, cx, cy

def cam_world_from_vtk(cam_tuple):
    pos, focal, up = [np.asarray(v, float) for v in cam_tuple]

    fwd = (focal - pos)
    fwd /= np.linalg.norm(fwd) + 1e-12
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-12
    up_cam = np.cross(right, fwd)

    # Camera axes as columns → world→camera
    R_cw = np.stack([right, up_cam, fwd], axis=1)   # (3×3)
    t_cw = -R_cw.T @ pos                            # note transpose here
    return R_cw.T, t_cw

def cam_T_obj(R_obj_w, t_obj_w, cam_tuple):
    R_w2c, t_w2c = cam_world_from_vtk(cam_tuple)
    R_co = R_w2c @ R_obj_w
    t_co = R_w2c @ t_obj_w + t_w2c
    return R_co, t_co

def actor_mesh(mesh, R, t):
    m = mesh.copy()
    T = np.eye(4)
    T[:3,:3] = R; T[:3,3] = t
    m.transform(T, inplace=True)
    return m

# ------------------- BOUNDS FROM DATA -------------------
tvecs = np.array([s["tvec"] for s in SAMPLES], dtype=float)
t_min, t_max = tvecs.min(axis=0), tvecs.max(axis=0)

rotvecs = np.array([scipyR.from_matrix(np.asarray(s["rotMat"])).as_rotvec()
                    for s in SAMPLES], dtype=float)
r_min, r_max = rotvecs.min(axis=0), rotvecs.max(axis=0)

# FOCAL BOUNDS (from the middle triplet only)
focals = np.array([s["camera_position"][1] for s in SAMPLES], dtype=float)
f_min, f_max = focals.min(axis=0), focals.max(axis=0)

# CAMERA DISTANCE STATS (pos <-> focal)
dists = np.array([np.linalg.norm(np.array(s["camera_position"][0]) - np.array(s["camera_position"][1]))
                  for s in SAMPLES], dtype=float)
dist_mean, dist_std = dists.mean(), dists.std()
d_min, d_max = dists.min(), dists.max()

# Optional padding
PAD_T = np.array([0.5, 0.5, 1.0])
PAD_R = np.deg2rad([5, 5, 5])
PAD_F = np.array([5.0, 5.0, 5.0])  # focal padding
PAD_D = 2.0                        # distance padding

t_low, t_high = t_min - PAD_T, t_max + PAD_T
r_low, r_high = r_min - PAD_R, r_max + PAD_R
f_low, f_high = f_min - PAD_F, f_max + PAD_F
d_low, d_high = max(d_min - PAD_D, 1e-3), d_max + PAD_D

# ------------------- SAMPLING HELPERS -------------------
def sample_uniform(low, high):
    return low + np.random.rand(*low.shape) * (high - low)


def add_key(plotter, key, func):
    if hasattr(plotter, "add_key_event_handler"):
        plotter.add_key_event_handler(key, func)
    else:
        plotter.add_key_event(key, func)

# ---------- TOP-VIEW CAMERA SAMPLER ----------
def _norm(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v/(n+1e-12)

def camera_forward_from_tuple(cam):
    pos, foc, _up = [np.asarray(x, float) for x in cam]
    return _norm(foc - pos)  # world forward

def sample_camera_top(up_axis=WORLD_UP, max_tilt_deg=MAX_TILT_DEG, max_roll_deg=MAX_ROLL_DEG):
    """
    Returns a PyVista camera_position tuple (pos, focal, up)
    that looks almost straight down (top view).
    """
    # 1) sample focal inside bounds
    focal = sample_uniform(f_low, f_high)

    # 2) pick a distance
    #    (use truncated normal around mean, clipped to [d_low, d_high])
    d = np.clip(np.random.normal(dist_mean, max(1e-3, dist_std)), d_low, d_high)

    # 3) forward vector: nearly -up_axis, with small tilt + random azimuth
    #    Build orthonormal basis around up_axis
    up = _norm(up_axis)
    # choose any vector not parallel to up to build basis
    tmp = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _norm(np.cross(tmp, up))     # axis 1 (in ground plane)
    y = _norm(np.cross(up, x))       # axis 2 (in ground plane)

    # azimuth around up, and small tilt away from straight-down
    az = np.random.uniform(0.0, 2.0*np.pi)
    tilt = np.deg2rad(np.random.uniform(0.0, max_tilt_deg))

    # Forward pointing DOWN: start at -up, add small tilt towards the ground plane
    ground_dir = np.cos(az) * x + np.sin(az) * y
    forward = _norm(-up*np.cos(tilt) + ground_dir*np.sin(tilt))

    # 4) camera position: pos = focal - forward*d
    pos = focal - forward * d

    # 5) view_up: start from 'y' axis in the image plane and apply small roll
    roll = np.deg2rad(np.random.uniform(-max_roll_deg, max_roll_deg))
    # camera right = normalize(cross(forward, up)) (points to image +x)
    right = _norm(np.cross(forward, up))
    # image up = cross(right, forward)
    img_up = _norm(np.cross(right, forward))
    # apply roll around 'forward'
    R_roll = scipyR.from_rotvec(roll * forward).as_matrix()
    view_up = _norm(R_roll @ img_up)

    return (tuple(pos), tuple(focal), tuple(view_up))


def render_and_augment_current_frame(plotter, out_path_png, bg_img,out_path_bg):
    """
    Grabs the current PyVista render (with your random background already set),
    applies white balrender_and_augment_current_frameance, vignette, light defocus blur, and specular flares,
    then saves the augmented image to out_path_png.
    """
    # 1) render & grab RGB (no transparency needed, bg is already in-scene)
    rgb = plotter.screenshot(transparent_background=False, return_img=True)  # HxWx3 uint8
    img = Image.fromarray(rgb).convert("RGB")
    bg = bg_img
    # 2) apply perturbations
    if random.random() < 0.8:
        img,bg = _apply_white_balance(img,bg, gain_range=(0.85, 1.50))
        #bg = _apply_white_balance(bg_img, gain_range=(0.85, 1.15))
    if random.random() < 0.5:
        img,bg = _apply_vignette(img,bg, strength_range=(0.05, 0.15))
        #bg = _apply_vignette(bg_img, strength_range=(0.05, 0.80))
    if random.random() < 0.5:
        img,bg = _apply_defocus_blur(img,bg, radius_range=(0.5, 5))
        #bg = _apply_defocus_blur(bg_img, radius_range=(0.5, 3))
    if random.random() < 0.5:
        img,bg = _add_specular_flares(img,bg, count_range=(1, 15), radius_range=(5, 20), opacity_range=(0.1, 1.0))
        #bg = _add_specular_flares(bg_img, count_range=(1, 10), radius_range=(5, 10), opacity_range=(0.1, 0.8), r = r_)
    if random.random() < 0.5:
        img,bg = _apply_gaussian_noise(img, bg, sigma_range=(5, 20))
    if random.random() < 0.5:   
        img,bg = _apply_color_jitter(img,bg)
    

    # 3) save
    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
    img.save(out_path_png)
    bg.save(out_path_bg)
    print(f"[INFO] Augmented frame saved to {out_path_png}")


# ------------------- INTERACTION -------------------
class State:
    actor: Optional[object] = None
    idx: int = 0
    R: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    cam: Optional[tuple] = None

state = State()

def nudge_bbox_into_view(R_obj_w, t_obj_w, cam_tuple, fx, fy, cx, cy, w, h,
                         max_iters=6, step_px=80, z_pad=0.02, znear=0.01):
    R_cw, t_cw = cam_world_from_vtk(cam_tuple)
    R_wc = R_cw.T
    t_w = t_obj_w.copy()

    for _ in range(max_iters):
        # project bbox corners
        Cw = (R_obj_w @ CORNERS_W.T).T + t_w[None,:]
        Cc = (R_cw @ Cw.T).T + t_cw[None,:]
        z = Cc[:,2]

        # ensure some positive depth; push forward if needed
        if np.min(z) <= znear + 1e-6:
            dz = (znear + z_pad) - np.min(z)
            t_w = t_w + R_wc @ np.array([0.0, 0.0, dz])
            continue

        u = fx * (Cc[:,0]/z) + cx
        v = fy * (Cc[:,1]/z) + cy

        # compute how much leaks past each border (positive = leaked)
        leak_left   = max(0.0,  np.max(0.0 - u))
        leak_right  = max(0.0,  np.max(u - (w - 1)))
        leak_top    = max(0.0,  np.max(0.0 - v))
        leak_bottom = max(0.0,  np.max(v - (h - 1)))

        if leak_left==leak_right==leak_top==leak_bottom==0.0:
            return t_w, True

        # desired pixel correction (sign so +dx moves bbox rightwards on screen)
        dx_px = leak_left - leak_right
        dy_px = leak_top  - leak_bottom

        # clamp so we don't overshoot
        mag = (dx_px**2 + dy_px**2) ** 0.5
        if mag > step_px:
            s = step_px / (mag + 1e-9)
            dx_px *= s; dy_px *= s

        # convert px → camera-metric at median depth (more stable than per-corner)
        z_med = float(np.median(z))
        dX = (dx_px / fx) * z_med
        dY = (dy_px / fy) * z_med

        # apply in camera X/Y, then back to world
        delta_cam = np.array([dX, dY, 0.0])
        t_w = t_w + R_wc @ delta_cam

    return t_w, all_corners_in_view(R_obj_w, t_w, cam_tuple, fx, fy, cx, cy, w, h, znear)


def all_corners_in_view(R_obj_w, t_obj_w, cam_tuple, fx, fy, cx, cy, w, h, znear=0.01, margin=4):
    # world->cam
    R_cw, t_cw = cam_world_from_vtk(cam_tuple)
    # object->world corners
    Cw = (R_obj_w @ CORNERS_W.T).T + t_obj_w[None,:]  # (8,3)
    # world->camera
    Cc = (R_cw @ Cw.T).T + t_cw[None,:]               # (8,3)
    z = Cc[:,2]
    if np.any(z <= znear + 1e-9):            # any corner behind/at near plane
        return False
    u = fx * (Cc[:,0]/z) + cx
    v = fy * (Cc[:,1]/z) + cy
    return (
        np.all(u >= margin) & np.all(u < (w - margin)) &
        np.all(v >= margin) & np.all(v < (h - margin))
    )

def _project_bbox_uv(R, t, cam, fx, fy, cx, cy):
    R_cw, t_cw = cam_world_from_vtk(cam)
    Cw  = (R @ CORNERS_W.T).T + t[None, :]
    Cc  = (R_cw @ Cw.T).T + t_cw[None, :]
    z   = Cc[:, 2]
    u   = fx * (Cc[:, 0] / z) + cx
    v   = fy * (Cc[:, 1] / z) + cy
    return float(u.min()), float(u.max()), float(v.min()), float(v.max()), float(z.min()), float(np.median(z))


def sample_visible_pose_guaranteed(plotter, cam, fx, fy, cx, cy,
                                   max_pose_tries=400,
                                   nudge_step_px=90,
                                   z_pad=0.02,
                                   znear=0.01,
                                   margin=6,
                                   # size floors relative to 800x544
                                   min_diag_frac=0.10,        # >= 10% of image diagonal
                                   min_area_frac=0.018,       # >= 1.8% of image area
                                   zoom_in_step=0.87,
                                   zoom_out_step=1.10,
                                   size_refine_iters=8):
    """
    Math-only guarantee: inside frustum + not tiny (relative to image size).
    Returns: R, t, cam (cam may be dolly-adjusted)
    """
    img_diag = (w**2 + h**2) ** 0.5
    min_diag_px = min_diag_frac * img_diag             # ~ 0.10 * 960 ≈ 96 px
    min_area_px = min_area_frac * (w * h)              # ~ 0.018 * 435k ≈ 7.8k px

    tries = 0
    forward_world = camera_forward_from_tuple(cam)

    while tries < max_pose_tries:
        tries += 1

        # --- sample pose
        t  = sample_uniform(t_low, t_high)
        rv = sample_uniform(r_low, r_high)
        R  = scipyR.from_rotvec(rv).as_matrix()

        # upright + facing camera (cheap early rejects)
        n_world = R @ MODEL_UP; n_world /= (np.linalg.norm(n_world) + 1e-12)
        if float(np.dot(n_world, WORLD_UP)) < MIN_UP_DOT:
            continue
        if float(np.dot(n_world, -forward_world)) <= 0.0:
            continue

        # push entire bbox in front of near plane if needed
        R_cw, t_cw = cam_world_from_vtk(cam); R_wc = R_cw.T
        u0,u1,v0,v1,zmin,_ = _project_bbox_uv(R, t, cam, fx, fy, cx, cy)
        if zmin <= znear + z_pad:
            t = t + R_wc @ np.array([0.0, 0.0, (znear + z_pad) - zmin])

        # ensure in-frame (nudge → optional zoom-out)
        if not all_corners_in_view(R, t, cam, fx, fy, cx, cy, w, h, znear, margin=margin):
            t_fix, ok = nudge_bbox_into_view(R, t, cam, fx, fy, cx, cy, w, h,
                                             max_iters=6, step_px=nudge_step_px, z_pad=z_pad, znear=znear)
            if ok:
                t = t_fix
            else:
                # mild zoom-out then recheck
                pos, focal, up = [np.asarray(v, float) for v in cam]
                dir_vec = focal - pos; dist = np.linalg.norm(dir_vec) + 1e-12
                fwd_cam = dir_vec / dist
                cam_try = (tuple(focal - fwd_cam * (dist * zoom_out_step)), tuple(focal), tuple(up))
                if all_corners_in_view(R, t, cam_try, fx, fy, cx, cy, w, h, znear, margin=margin):
                    cam = cam_try
                else:
                    continue  # resample

        # size floor (bbox diag + area)
        u0,u1,v0,v1,_,_ = _project_bbox_uv(R, t, cam, fx, fy, cx, cy)
        bw, bh = max(0.0, u1-u0), max(0.0, v1-v0)
        diag   = float(np.hypot(bw, bh))
        area   = float(bw * bh)

        if (diag < min_diag_px) or (area < min_area_px):
            # iterative zoom-in with safety checks
            for _ in range(size_refine_iters):
                pos, focal, up = [np.asarray(v, float) for v in cam]
                dir_vec = focal - pos; dist = np.linalg.norm(dir_vec) + 1e-12
                fwd_cam = dir_vec / dist
                cam_try = (tuple(focal - fwd_cam * (dist * zoom_in_step)), tuple(focal), tuple(up))

                # keep plane-safe
                u0,u1,v0,v1,zmin,_ = _project_bbox_uv(R, t, cam_try, fx, fy, cx, cy)
                if zmin <= znear + z_pad:
                    break  # too close

                # keep framed
                if not all_corners_in_view(R, t, cam_try, fx, fy, cx, cy, w, h, znear, margin=margin):
                    t_fix, ok = nudge_bbox_into_view(R, t, cam_try, fx, fy, cx, cy, w, h,
                                                     max_iters=6, step_px=nudge_step_px, z_pad=z_pad, znear=znear)
                    if ok:
                        t = t_fix; cam = cam_try
                    else:
                        break
                else:
                    cam = cam_try

                # re-measure
                u0,u1,v0,v1,_,_ = _project_bbox_uv(R, t, cam, fx, fy, cx, cy)
                bw, bh = max(0.0, u1-u0), max(0.0, v1-v0)
                diag   = float(np.hypot(bw, bh))
                area   = float(bw * bh)
                if (diag >= min_diag_px) and (area >= min_area_px):
                    break

            if (diag < min_diag_px) or (area < min_area_px):
                continue  # resample

        return R, t, cam

    # fallback (rare)
    return R, t, cam

def preflight_depth_gate(plotter, R, t, cam, min_fg_px=6000):
    """
    Temporarily add a white, unlit actor; render once; count depth pixels; remove actor.
    Returns True if foreground coverage >= min_fg_px.
    """
    posed = actor_mesh(mesh, R, t)
    temp_actor = plotter.add_mesh(
        posed,
        color=(255,255,255),
        pbr=False,
        smooth_shading=False,
        reset_camera=False,
        lighting=False,      # emissive
        opacity=1.0
    )
    old_cam = plotter.camera_position
    plotter.camera_position = cam
    plotter.render()

    depth = plotter.get_image_depth()
    fg = int(np.isfinite(depth).sum())

    plotter.remove_actor(temp_actor, render=False)
    plotter.camera_position = old_cam
    plotter.render()
    return fg >= min_fg_px


def on_new_pose2():
    cam = sample_camera_top()
    forward = camera_forward_from_tuple(cam)

    try:
        plotter.remove_background_image()
    except Exception:
        pass
    bg_path = random.choice(BACKGROUND_IMAGES)
    plotter.add_background_image(os.path.join('D:\\Harvard\\implementations\\GT_Generator\\Data\\noclips\\', bg_path))

    tries = 0
    fx, fy, cx, cy = intrinsics_from_vtk(plotter.camera, w, h)  # fov-only; pose-independent
    
    #for _ in range(100):
    R, t, cam = sample_visible_pose_guaranteed(
            plotter, cam, fx, fy, cx, cy,
            max_pose_tries=200, nudge_step_px=90, z_pad=0.02, znear=znear,
            min_diag_frac=0.10, min_area_frac=0.018, size_refine_iters=8,
            zoom_in_step=0.87, zoom_out_step=1.10, margin=6
        )
        #plotter.show(auto_close=False, interactive_update=True)
        #if preflight_depth_gate(plotter, R, t, cam, min_fg_px=6000):
        #    break  # guaranteed to show up

    # remove previous actor if any
    if state.actor is not None:
        plotter.remove_actor(state.actor, render=False)

    posed = actor_mesh(mesh, R, t)
    state.actor = plotter.add_mesh(
        posed,
        color="#d9dde2",
        pbr=True,
        metallic=1.0,
        roughness=0.30,
        specular=1.0,
        diffuse=1.0,
        smooth_shading=True,
        reset_camera=False,
        ambient=0.1)

    state.R = R
    state.t = t
    state.cam = cam
    plotter.camera_position = cam

    plotter.render()

    print("\n--- RANDOM TOP VIEW (upright) ---")
    print("tvec:", np.round(t, 3))
    #print("rotvec(rad):", np.round(rv, 4))
    print("rotMat:\n", np.round(R, 2))
    print("camera_position:", plotter.camera_position)


# def on_save():
#     state.idx += 1
#     rgb_path  = os.path.join(OUT_DIR, f"{state.idx:03d}.png")
#     bg_path  = os.path.join(OUT_DIR, f"{state.idx:03d}_bg.png")
#     mask_path = os.path.join(OUT_DIR, f"{state.idx:03d}_mask.png")

#     #---- save background
#     state.actor.SetVisibility(False)   # hide
#     plotter.render()
#     bg = plotter.screenshot(transparent_background=False, return_img=True)  # HxWx3 uint8
#     bg_ = Image.fromarray(bg).convert("RGB")
#     state.actor.SetVisibility(True)    # show again
#     plotter.render()
#     # --- keep current setup
#     bg_backup = plotter.renderer.GetBackground()
#     saved_lights = list(plotter.renderer.lights)      # <— save refs

#     # RGB (normal)
#     #render_and_augment_current_frame(plotter, rgb_path,bg_,bg_path)
#     bg_.save(bg_path)
#     plotter.screenshot(rgb_path, transparent_background=False)

#     # MASK (black bg, no lights)
#     plotter.set_background("black")
#     plotter.renderer.remove_all_lights()
#     plotter.remove_background_image()
#     plotter.render()
#     plotter.screenshot(mask_path, transparent_background=True)
#     mask = convert_to_binary_mask(image_path = mask_path,output_path = mask_path)
#     plotter.set_background(bg_backup)
#     for L in saved_lights:
#         plotter.add_light(L)
#     plotter.render()
    
#     assert state.R is not None and state.t is not None and state.cam is not None
#     json_path = os.path.join(OUT_DIR, f"{state.idx:03d}.json")
#     save_pose_and_camera(state.t, state.R, state.cam, plotter.camera, json_path)
    

#     print(f"Saved: {rgb_path} | {mask_path}")
#     return mask

def on_save():
    state.idx += 1
    stem = f"{state.idx:03d}"
    rgb_path  = os.path.join(OUT_DIR, f"{stem}.png")
    bg_path   = os.path.join(OUT_DIR, f"{stem}_bg.png")
    mask_path = os.path.join(OUT_DIR, f"{stem}_mask.png")

    # save BG
    state.actor.SetVisibility(False); plotter.render()
    bg = plotter.screenshot(transparent_background=False, return_img=True)
    Image.fromarray(bg).convert("RGB").save(bg_path)
    state.actor.SetVisibility(True); plotter.render()

    # save RGB
    plotter.screenshot(rgb_path, transparent_background=False)

    # save MASK (black bg, no lights)
    bg_backup = plotter.renderer.GetBackground()
    saved_lights = list(plotter.renderer.lights)
    plotter.set_background("black")
    plotter.renderer.remove_all_lights()
    plotter.remove_background_image()
    plotter.render()
    plotter.screenshot(mask_path, transparent_background=True)
    convert_to_binary_mask(mask_path, mask_path)
    # restore
    plotter.set_background(bg_backup)
    for L in saved_lights: plotter.add_light(L)
    plotter.render()

    # now save JSON (image files exist)
    json_path = os.path.join(OUT_DIR, f"{stem}.json")
    save_pose_and_camera(state.t, state.R, state.cam, plotter.camera, json_path,
                         w=w, h=h, znear=znear, zfar=zfar,
                         files={"rgb": os.path.basename(rgb_path),
                                "bg": os.path.basename(bg_path),
                                "mask": os.path.basename(mask_path)})



def convert_to_binary_mask(image_path, output_path="binary_mask.png", threshold=10):
    """
    Convert an image with black background and white/gray annotation
    into a binary mask (0=background, 255=foreground).

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the binary mask.
        threshold (int): Pixel intensity threshold (0-255).
                         Any pixel above this is considered foreground.

    Returns:
        mask (numpy.ndarray): Binary mask (0 and 255).
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold: anything > threshold becomes 255 (foreground)
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Save result
    cv2.imwrite(output_path, mask)

    return mask


def _rotmat_to_quat_wxyz(R: np.ndarray):
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    q /= (np.linalg.norm(q) + 1e-12)
    if q[0] < 0:  # ensure w ≥ 0
        q = -q
    return q.tolist()


def save_pose_and_camera(tvec_obj_w, R_obj_w, camera_position, plotter_camera,
                         out_path, w, h, znear, zfar, files):
    # object→camera
    R_co, t_co = cam_T_obj(R_obj_w, tvec_obj_w, camera_position)
    fx, fy, cx, cy = intrinsics_from_vtk(plotter_camera, w, h)

    data = {
        "image": {
            "file_rgb": files["rgb"],
            "file_bg": files["bg"],
            "file_mask": files["mask"],
            "height": int(h), "width": int(w)
        },
        "camera": {
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "znear": float(znear), "zfar": float(zfar),
            "position": list(map(float, camera_position[0])),
            "focal_point": list(map(float, camera_position[1])),
            "view_up": list(map(float, camera_position[2]))
        },
        "clip": {
            "pose_se3": {
                "quaternion_wxyz": _rotmat_to_quat_wxyz(R_co),
                "translation_m": list(map(float, t_co))
            },
            "pose_world": {
                "quaternion_wxyz": _rotmat_to_quat_wxyz(R_obj_w),
                "translation_m": list(map(float, tvec_obj_w))
            }
        },
        "conventions": {
            "pixel_origin": "top_left",
            "pixel_center": 0.5,
            "handedness": "right",
            "cam_forward_axis": "-Z"
        }
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)




def mask_is_black(img, min_fg_px=1, thresh=1):
    """Treats values > thresh as foreground. Returns True if 'black'."""
    #img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:  # RGB/RGBA -> grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fg = cv2.countNonZero((img > thresh).astype(np.uint8))
    return fg < min_fg_px  # e.g., 1 means 'no foreground at all'

N = 500
# Start with one pose and show
on_new_pose2()
plotter.show(auto_close=False, interactive_update=True)
#state.idx += 3390
for i in range(N):
    plotter.update()
    on_new_pose2()
    mask = on_save()
    #if cv2.countNonZero(mask) == 0:
    #    state.idx = state.idx - 1

plotter.close()