# Python environment used for testing

Conda environment `bn4601`: Python 3.11.16, PyTorch 2.11.0 with CUDA 12.8, Windows 11, NVIDIA RTX 4050 6 GB.
`pip_freeze_bn4601.txt` lists the exact package versions.

To rebuild it:

```bash
conda create -n bn4601 python=3.11 -y
conda activate bn4601
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r ../app_snapshot/app/requirements.txt accelerate bitsandbytes
```

Use the freeze file if a newer package breaks something. `scikit-learn==1.9.1` is required by the saved vitals model.
