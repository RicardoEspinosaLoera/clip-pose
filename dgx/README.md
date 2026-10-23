# Ejecución del pipeline en la DGX

Pipeline completo del artículo (IPCAI 2027, envío 30 oct 2026): generación sintética con la receta final → validación del GT → splits → entrenamientos (E0, E1, E2, E5) → evaluación sintética y real → tabla de resultados.

Todo corre desde esta carpeta con rutas relativas. No hace falta pantalla: VTK dibuja en un X virtual (Xvfb + Mesa, CPU); las GPUs las usan GroundingDINO (evitar instrumentos) y PyTorch (entrenar/evaluar).

## 0. Qué hay en el paquete

| Carpeta | Contenido | Tamaño |
| --- | --- | --- |
| `GT_Generator/` | generador, apariencia, validadores, mallas (`clip_lazic_std.obj` + `.parts.json` = L-Clip de MACS, `Item.obj` legado), `keypoints*.json` | 30 MB |
| `GT_Generator/Data/noclips_bank/` | 600 fondos reales sin clip (MACS) | 252 MB |
| `pose_train/` | dataset, modelos M1/M2, train, splits, eval sintética y real | < 1 MB |
| `Inpaint_Realism/lora/clean_clips2(_masks)/` | recortes reales de clip para la métrica de brecha de dominio (KID) | 476 MB |
| `real_eval/` | 200 frames reales seleccionados (14 videos) + `frames.json`; `annotations/<nombre>/` cuando existan | 25 MB |
| `models/hf/` | GroundingDINO tiny en formato caché de Hugging Face | 659 MB |
| `models/torch/` | ResNet-34 y ResNet-18 ImageNet (torchvision) | 130 MB |
| `dgx/` | esta carpeta: Dockerfile, entorno, scripts `00`–`50` | — |

Con `--with-repaint` el paquete añade SDXL inpainting + VAE + ControlNet canny (~22 GB) y el LoRA MACS, solo para la variante A4. Si la DGX tiene internet, es mejor descargarlos allá (ver sección 7).

## 1. Llevar el código y los datos a la DGX

Opción git (la normal): el repositorio `https://github.com/RicardoEspinosaLoera/clip-pose` contiene el código, la malla, los 600 fondos, los recortes reales y los frames de evaluación. Los pesos (GroundingDINO, ResNet ImageNet) se descargan solos la primera vez si el contenedor tiene internet.

```bash
cd /workspace/clip-pose && git pull        # o: git clone https://github.com/RicardoEspinosaLoera/clip-pose.git
```

Opción tar (sin internet en la DGX, o para llevar también los pesos): en la laptop

```bash
python dgx/make_bundle.py                 # ~1.6 GB -> C:\Users\ricar\realac_bundle\realac_dgx_bundle_<fecha>.tar
python dgx/make_bundle.py --with-repaint  # +22 GB (opcional)
scp C:/Users/ricar/realac_bundle/realac_dgx_bundle_<fecha>.tar usuario@dgx:/raid/
```

En la DGX:

```bash
cd /raid && tar xf realac_dgx_bundle_<fecha>.tar && cd RealAC_Generator
```

## 2. Entorno

Opción A, Docker (recomendada):

```bash
docker build -t realac:latest dgx/
docker run --gpus all --shm-size=32g -v /raid/RealAC_Generator:/work -w /work -it realac:latest bash
```

Opción B, conda sobre el sistema (requiere `xvfb`, `libgl1-mesa-dri`, `libglx-mesa0`, `libegl1` instalados por el administrador):

```bash
bash dgx/install_conda.sh realac && conda activate realac
```

Opción C, contenedor ya existente (por ejemplo `root@b87d52538c1e:/workspace/clip-pose`): desde el host, `docker cp realac_dgx_bundle_<fecha>.tar b87d52538c1e:/workspace/clip-pose/`, y dentro del contenedor:

```bash
cd /workspace/clip-pose && tar xf realac_dgx_bundle_<fecha>.tar --strip-components=1
bash dgx/setup_container.sh                      # apt: xvfb + mesa; si la imagen ya trae torch con CUDA (NGC) lo conserva, si no crea .venv con torch cu128; pruebas
# USE_SYSTEM_TORCH=1 / FORCE_VENV=1              # fuerzan una de las dos rutas
```

Si la instalación de torch falla con `from versions: none`, el script imprime la versión de Python, la arquitectura y si llega a download.pytorch.org y pypi.org: sin internet hay que usar una imagen con torch incluido (`nvcr.io/nvidia/pytorch`) o configurar `HTTPS_PROXY`; con internet pero sin rueda, la Python del contenedor no está soportada (usar 3.10–3.13).

Si `/dev/shm` es pequeño (el script lo imprime), reiniciar el contenedor con `--ipc=host` o entrenar con `WORKERS=0`.

Comprobar: `nvidia-smi` (driver ≥ 525 para las ruedas cu128; si es más viejo, reinstalar torch con `--index-url https://download.pytorch.org/whl/cu126`) y `python -c "import torch; print(torch.cuda.is_available())"`.

## 3. Orden de ejecución

```bash
bash dgx/00_smoke.sh                         # 10-20 min: 10 muestras, validación, 60 iteraciones, evaluación. Imprime s/muestra/shard
N_TOTAL=5000 SHARDS=16 bash dgx/10_generate.sh   # set final en GT_Generator/out_v3 (stems 00001..05000) + validación
bash dgx/20_splits.sh                        # train/val/test por fondo (80/10/10) + subconjuntos 1 250 y 2 500
bash dgx/30_train.sh                         # E0 (M1, M2), E1 (1 250, 2 500), E2 A0 (render plano): 5 corridas, una por GPU
bash dgx/40_eval.sh                          # test sintético de cada corrida (+ frames reales si hay anotaciones) -> pose_train/runs/results.md
```

Ablaciones de apariencia (E2 del protocolo, mismas poses y semillas). A0 no necesita set aparte (el generador guarda `NNNNN_flat.png`) y A4 es el propio set de E0:

```bash
bash dgx/50_variants.sh a2                   # A2: anodizado + IBL + óptica, sin huella de video ni colocador (el del plan reducido)
bash dgx/50_variants.sh a1                   # A1: anodizado + IBL solamente
bash dgx/50_variants.sh a3                   # A3: A2 + huella de video (= receta final sin colocador)
bash dgx/50_variants.sh a5                   # A5: colocador repintado con SDXL + LoRA (necesita los pesos SDXL, sección 7)
bash dgx/30_train.sh && bash dgx/40_eval.sh  # detecta los sets out_v3_a1 … out_v3_a5 y los entrena/evalúa (corridas e2_a1 … e2_a5)
SEEDS='1 2' bash dgx/30_train.sh             # semillas extra de E0 para la versión journal
```

Para las ablaciones de generación de la ronda journal (E3 tamaño del clip, E4 filtro de instrumentos) se vuelve a generar con otra carpeta: `DATA_OUT=GT_Generator/out_e3 CLIP_FRAC_MAX=0.003 bash dgx/10_generate.sh` y `DATA_OUT=GT_Generator/out_e4 USE_DINO_AVOID=0 BLADE_PROBE=0 bash dgx/10_generate.sh`.

Cada script se puede relanzar: `10_generate.sh` reescribe los mismos stems con las mismas semillas; `30_train.sh` salta las corridas con `best.pt` (`FORCE=1` para repetir); `40_eval.sh` salta las evaluaciones ya hechas.

## 4. Tiempos esperados

| Paso | Laptop (RTX 5060, 8 GB) | DGX (estimado, verificar con `00_smoke.sh`) |
| --- | --- | --- |
| Generar 5 000 muestras | 32 s/muestra con GPU ocupada, ~44 h | render por CPU (Mesa): 30–60 s/muestra/shard; con 16 shards 1,5–3 h |
| Validación del set (fidelidad, instrumentos, KID) | 1 h | 20–30 min |
| Un entrenamiento, 30 000 iteraciones, lote 16 | 8 h | 1–1,5 h en A100/H100 (todas las corridas en paralelo) |
| Evaluación por corrida | minutos | minutos |

## 5. Qué devolver

- `pose_train/runs/results.md` y `results.csv` (una fila por corrida) y `results_subgroups.csv`.
- `pose_train/runs/<corrida>/`: `config.json`, `log.csv`, `best.pt`, `test/summary.json`, `test/per_sample.csv`, `real/` si hubo anotaciones.
- `GT_Generator/out_v3/_pose_fidelity.csv`, `_pose_fidelity.jpg`, `_tool_clearance.jpg` y la salida de `domain_gap.py` (en `logs/`).
- `logs/` completo.

## 6. Variables útiles (todas con valor por defecto en `dgx/env.sh`)

| Variable | Uso | Por defecto |
| --- | --- | --- |
| `DATA_OUT` | carpeta del set de 5 000 | `GT_Generator/out_v3` |
| `RUNS`, `LOGS` | salidas de entrenamiento y logs | `pose_train/runs`, `logs` |
| `N_TOTAL`, `SHARDS`, `BASE_SEED` | tamaño, procesos y semilla de la generación | 5000, 16, 1000 |
| `ITERS`, `BATCH`, `WORKERS` | entrenamiento | 30000, 16, 8 |
| `ONLY`, `SEEDS`, `FORCE`, `NOWAIT` | filtro regex de corridas, semillas extra, repetir, no esperar | — |
| `ANNOTATOR` | `real_eval/annotations/<nombre>/` para la evaluación real | `ricardo` |
| `HF_HUB_OFFLINE` | 1 = usar solo `models/hf`; 0 = permitir descargas | 1 |
| `NGPU`, `LP_NUM_THREADS`, `OMP_NUM_THREADS` | GPUs a repartir, hilos de render por shard | detectado, 8, 8 |

Variables de la receta del generador (`TOOL_PROB`, `CLIP_FRAC_*`, `AVOID_*`, `DINO_*`, …) están escritas en `10_generate.sh` y coinciden con los valores por defecto del script; no cambiarlas para el artículo.

## 7. Problemas conocidos

- Driver viejo en el host (`nvidia-smi` cabecera: p. ej. 465.19, CUDA 11.3): las ruedas cu12x no cargan. `setup_container.sh` lo detecta y usa torch 2.6.0 con CUDA 11.8, la última build cu118 (funciona en drivers ≥ 450 por compatibilidad de versión menor); el resto de dependencias es el mismo.
- Imagen vieja con Python 3.8 (p. ej. torch 1.12 + cu113): `setup_container.sh` instala Miniforge en `/opt/miniforge` con un entorno `realac` (Python 3.10) y los scripts lo usan solos. Alternativa: contenedor nuevo desde `pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel` (driver viejo) o `nvcr.io/nvidia/pytorch:25.04-py3` (driver ≥ 525), clonar dentro y correr el setup.

- `start_xvfb failed` o ventana VTK vacía: falta `xvfb` o Mesa; en Docker ya vienen. Alternativa sin X: `pip install --extra-index-url https://wheels.vtk.org vtk-osmesa` si existe la versión 9.6.
- `DataLoader worker killed`: subir `--shm-size` del contenedor o bajar `WORKERS`.
- Errores de Hugging Face sin red: el paquete incluye GroundingDINO en `models/hf`; con `HF_HUB_OFFLINE=1` no se intenta descargar nada. Para A4 con internet: `HF_HUB_OFFLINE=0 huggingface-cli download diffusers/stable-diffusion-xl-1.0-inpainting-0.1` y lo mismo para `madebyollin/sdxl-vae-fp16-fix` y `diffusers/controlnet-canny-sdxl-1.0`, con `HF_HOME` apuntando a `models/hf`.
- `train.py` no reanuda: si una corrida muere, relanzar con `ONLY=<nombre> FORCE=1`.
- La evaluación real solo aplica a M1 y requiere `real_eval/annotations/<nombre>/*.json` generados con `pose_train/annotate_tool.py` en la laptop; copiarlos a la DGX antes de `40_eval.sh`.
- Los stems del set tienen cinco dígitos (`00001`); los sets antiguos de la laptop (`001`) siguen funcionando con los mismos scripts.
