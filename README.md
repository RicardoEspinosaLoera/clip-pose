# clip-pose

Estimación de pose 6-DoF de clips de aneurisma a partir de datos sintéticos (PyVista/VTK) con apariencia MACS, evaluada en frames reales. Protocolo y plan de publicación (IPCAI 2027 / IJCARS) en el documento del proyecto.

## Estructura

| Carpeta | Qué es |
| --- | --- |
| `GT_Generator/` | generador de muestras sintéticas con GT exacto (`render_clip_poses_depth.py`), apariencia y colocador (`restyle_appearance.py`), validadores (`fix_pose_se3.py`, `measure_pose_fidelity.py`, `check_tool_clearance.py`, `domain_gap.py`), malla `Item.obj`, `keypoints.json`, 600 fondos reales en `Data/noclips_bank/` |
| `pose_train/` | dataset, modelos M1 (puntos clave + PnP) y M2 (regresión directa), `train.py`, splits, `eval_synth.py`, `eval_real.py`, herramientas de selección y anotación de frames reales |
| `real_eval/` | 200 frames reales seleccionados (`frames.json`) y anotaciones por anotador |
| `Inpaint_Realism/` | ruta de realismo por difusión (SDXL + LoRA MACS): scripts, recortes reales de clip (`lora/clean_clips2`) para la métrica de brecha de dominio |
| `dgx/` | ejecución completa en la DGX: entorno, generación por shards, splits, entrenamientos, evaluación, tabla de resultados. **Empezar por [dgx/README.md](dgx/README.md)** |
| `src/`, `scritps/`, `meshes/`, `config.yaml` | código anterior del repositorio (entrenamiento con DINOv3/LoRA), sin cambios |

## Arranque rápido en la DGX

```bash
cd /workspace/clip-pose && git pull
bash dgx/setup_container.sh     # Xvfb + Mesa, venv con torch 2.11 cu128 y dependencias fijadas, pruebas
bash dgx/00_smoke.sh            # 10 muestras, validación, entrenamiento corto, evaluación
N_TOTAL=5000 SHARDS=16 bash dgx/10_generate.sh && bash dgx/20_splits.sh && bash dgx/30_train.sh && bash dgx/40_eval.sh
```

Pesos de modelos (`models/`), sets generados (`GT_Generator/out*`), corridas (`pose_train/runs/`) y logs no se versionan.
