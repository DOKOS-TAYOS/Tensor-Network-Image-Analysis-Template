# `tn_dl`

Plantilla base para proyectos de deep learning con tensor networks usando `torch` y `tensorkrowch`.

## Qué trae

- Estructura modular en `src/tn_dl`.
- Configuración con YAML y dataclasses tipadas.
- Baseline de clasificación de imágenes con `TrigPixelEmbedding` + `MPSClassifier`.
- Entrenamiento y evaluación con selección de dispositivo `auto | cpu | cuda`.
- Soporte para `MNIST` y `FashionMNIST`.
- Scripts para local y para envío a `SLURM`.

## Entorno

En Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

En Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Ejecución local

```powershell
.\scripts\train_local.ps1 -Config configs/experiments/mnist_mps.yaml -Runtime configs/runtime/local.yaml -Device auto
```

```bash
./scripts/train_local.sh configs/experiments/mnist_mps.yaml configs/runtime/local.yaml auto
```

## Evaluación

```powershell
.\.venv\Scripts\python.exe -m tn_dl.cli.eval --config configs/experiments/mnist_mps.yaml --runtime configs/runtime/local.yaml --checkpoint runs/<run>/best.pt
```

## SLURM

```bash
./scripts/submit_slurm.sh configs/experiments/mnist_mps.yaml configs/runtime/slurm.yaml
```

Puedes sobrescribir recursos con variables de entorno como `SLURM_PARTITION`, `SLURM_TIME`, `SLURM_MEMORY`, `SLURM_GPUS`, `SLURM_CPUS_PER_TASK` y `SLURM_ACCOUNT`.

## Calidad

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check . --fix
.\.venv\Scripts\python.exe -m ruff format .
```
