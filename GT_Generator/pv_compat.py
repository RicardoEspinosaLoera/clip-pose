"""
pv_compat.py -- let PyVista create off-screen Plotters on a host where Windows Application Control blocks
vtkInteractionWidgets.pyd (laptop, 25 Oct 2026: "Una directiva de Control de aplicaciones bloqueo este archivo";
the same policy blocks torchvision/_C.pyd, see _import_transformers_dino in render_clip_poses_depth.py).
PyVista's RenderWindowInteractor only touches that module for the context interactor style and the picker
table, neither of which an off-screen render uses. install() wraps pyvista._vtk's lazy attribute lookup so
a name whose VTK module cannot load resolves to an inert dummy class instead of raising. No-op where VTK is intact.
"""
import importlib

_BLOCKED = ("vtkInteractionWidgets", "vtkViewsCore", "vtkViewsContext2D")


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, k):
        return lambda *a, **kw: None


def install():
    try:
        importlib.import_module("vtkmodules.vtkInteractionWidgets")
        return False                                        # intact VTK: nothing to do
    except ImportError:
        pass
    V = importlib.import_module("pyvista._vtk")
    orig = getattr(V, "__getattr__", None)
    if orig is None or getattr(V, "_pv_compat_installed", False):
        return False

    def _getattr(name):
        try:
            return orig(name)
        except ImportError as e:
            if any(b in str(e) for b in _BLOCKED):
                return _Dummy
            raise

    V.__getattr__ = _getattr
    V._pv_compat_installed = True
    print("[pv_compat] vtkInteractionWidgets is blocked on this host: interactor style / pickers stubbed", flush=True)
    return True
