# Conda environment "mls" — OpenPCDet + torch.compile

Environment for running OpenPCDet with **torch.compile** (Python 3.11 + PyTorch 2.1+).

## Quick: one script

From **OpenPCDet** root:

```bash
bash create_mls_env.sh
```

Then activate and verify:

```bash
conda activate mls
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); from pcdet.config import cfg; print('pcdet ok')"
python tools/export.py --compile --output pointpillar_traced_compiled.pt
```

## Manual steps (if script fails or you prefer)

1. **Create env** (from OpenPCDet root):

   ```bash
   conda env create -f environment_mls.yml
   conda activate mls
   ```

2. **PyTorch 2.1 + CUDA 11.8** (required for torch.compile with Python 3.11):

   ```bash
   pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **spconv** (match CUDA; use cu116/cu117/cu120 if your driver differs):

   ```bash
   pip install spconv-cu118
   ```

4. **OpenPCDet deps and pcdet** (use `numpy<2` and `setuptools<70` for compatibility):

   ```bash
   pip install -r requirements.txt
   pip install 'numpy<2' 'setuptools>=58,<70'
   python setup.py develop
   ```

## Notes

- **CUDA**: If you use CUDA 12, install `spconv-cu120` and PyTorch with `cu121` index.
- **Recreate env**: `conda env remove -n mls` then run the script or manual steps again.
