import pyvista as pv
import numpy as np
from vtk import vtkMatrix4x4, vtkMatrix3x3
import cv2
from scipy.spatial.transform import Rotation as scipyR
np.set_printoptions(precision=2)


print("PRESS 'q' TO PROJECT MESH ON TEXTURE")

def arrayFromVTKMatrix(vmatrix):
    if isinstance(vmatrix, vtkMatrix4x4):
        matrixSize = 4
    elif isinstance(vmatrix, vtkMatrix3x3):
        matrixSize = 3
    else:
        raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    narray = np.eye(matrixSize)
    vmatrix.DeepCopy(narray.ravel(), vmatrix)
    return narray
   

w = 1024
h = 1024
znear = 0.1
zfar = 1000


MESH_PATH = 'ItemLowPoly.obj'
TEX_PATH = 'Data/clips/20.png' 


plotter = pv.Plotter(off_screen=False, window_size = [w, h])
plotter.add_background_image(TEX_PATH)

mesh = pv.read(MESH_PATH)

plotter.add_mesh(
    mesh,
    color="#d9dde2",     # lighter grey-blue albedo
    pbr=True,
    metallic=1.0,        # keep metallic
    roughness=0.30,      # 0.2–0.3 gives satin sheen
    smooth_shading=True,
)

class PlotterControls:
    def __init__(self, plotter):
        self.p = plotter
        self.p.add_key_event("q", self.quit)
        self.stop = False
        
    # Whenever 'q' is pressed, end program
    def quit(self):
        print("Stopped")
        self.stop = True

pc = PlotterControls(plotter)

plotter.camera.clipping_range = (znear, zfar)
#plotter.add_light(pv.Light(position=(2, 1, 3), light_type='scene light', intensity=1.0))
#plotter.add_light(pv.Light(position=(-3, -2, 1), light_type='scene light', intensity=0.6))
headlight = pv.Light(light_type='headlight', intensity=1.2, color='white')
plotter.add_light(headlight)
plotter.show(auto_close=False, interactive_update=True)

camera = plotter.camera
view_angle = camera.GetViewAngle()
focal_length = 0.5 * w / np.tan(np.radians(0.5 * view_angle))


while True:
    
    # get the modelview camera
    vmtx = plotter.camera.GetModelViewTransformMatrix()

    m = arrayFromVTKMatrix(vmtx)
    rotMat = np.array([
           [m[0][0], m[0][1], m[0][2]],
           [m[1][0], m[1][1], m[1][2]],
           [m[2][0], m[2][1], m[2][2]]
        ])

    tVec = np.array([m[0][3], m[1][3], m[2][3]])

    print('tvec------------')  
    print(tVec)    
    print('rotMat------------')  
    print(rotMat)
    print('camera_position------------')
    print(plotter.camera_position)

    
    plotter.update()  # Non-blocking call to render updated environment, this also catches the key events
    if pc.stop:
        plotter.close()
        break
