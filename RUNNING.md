# Running the YOLIC baseline vs YOLIC-Align experiments

Goal: train the original head (`baseline`) and the new head (`align`) on all three datasets with identical settings, evaluate both, and send back the numbers.

Nothing in this branch has been run yet, so start with the smoke test (step 2) before committing to long runs.

## 1. Setup

```
git clone https://github.com/Mooki123/YOLIC-Align.git
cd YOLIC-Align
git checkout fix/align-checkpoint-loading
pip install torch torchvision numpy opencv-python pandas scikit-learn matplotlib pillow
```

Use a CUDA build of PyTorch if you have a GPU (see pytorch.org). Training on CPU works but is very slow.

## 2. Smoke test (about a minute, no data needed)

```
python -c "import torch, yolic_align as y; [print(d, y.build_model('align', d, n, k+1)(torch.randn(2,3,224,224)).shape) for d,n,k in [('indoor',30,6),('outdoor',104,11),('cityscapes',256,3)]]"
```

Expected output:
```
indoor torch.Size([2, 210])
outdoor torch.Size([2, 1248])
cityscapes torch.Size([2, 1024])
```

If this fails, stop and send the error.

## 3. Data layout

Scripts read data from the folder you run them in. Indoor and Outdoor both use a folder called `images`, so give each its own working directory.

| Dataset | Run from a folder containing | Notes |
|---|---|---|
| Indoor | `images/`, `labels/` | 6410 frames |
| Outdoor | `images/`, `yoliclabel/` | 20,380 frames |
| Cityscapes | `Datasets/Cityscapes/leftImg8bit/` and `Datasets/Cityscapes/gtFine/` | standard Cityscapes download |

Simplest way: make three folders (`indoor_run/`, `outdoor_run/`, `cityscapes_run/`), put the data in each, and run the scripts from there with the full path to the script, e.g. `python ../indoor_yolic.py --arch align`. Python finds the helper modules automatically because they sit next to the script.

The train/val/test split is fixed (`random_state=2`), so every run uses the same split.

## 3b. Quick trial run (do this first)

Run one dataset for a single epoch per head to catch problems early:

```
python indoor_yolic.py --arch baseline --epochs 1
python indoor_yolic.py --arch align --epochs 1
```

Then delete the files it wrote (see step 4 for names) before starting the real runs.

## 4. Training

Each command trains for 150 epochs by default. Run every one of them for both heads:

```
python indoor_yolic.py --arch baseline
python indoor_yolic.py --arch align

python outdoor_yolic.py --arch baseline
python outdoor_yolic.py --arch align

python cityscapes_yolic.py --arch baseline
python cityscapes_yolic.py --arch align
```

Cityscapes is the slowest because its labels are built on the CPU from full-size images.

Each run writes, in the folder you ran it from, a checkpoint and a CSV of per-epoch results:

| Dataset | Baseline | Align |
|---|---|---|
| Indoor | `mobilenet_indoor.pth.tar` | `mobilenet_indoor_align.pth.tar` |
| Outdoor | `mobilenet_outdoor.pth.tar` | `mobilenet_outdoor_align.pth.tar` |
| Cityscapes | `cityscapes_mobilenet.pth.tar` | `cityscapes_mobilenet_align.pth.tar` |

The CSV has the same name with `.csv` instead of `.pth.tar`.

### More seeds (recommended)

Differences of about 1 F1 point are within noise, so if time allows repeat every run with `--seed 2` and `--seed 3`. The file names do not include the seed, so a new run overwrites the previous one. After each run, move the `.pth.tar` and `.csv` files into a folder (for example `seed1/`) before starting the next seed, then run e.g. `python indoor_yolic.py --arch align --seed 2`.

## 5. Evaluation

The eval scripts print precision, recall and F1. Pass `--weights` with the checkpoint path, because each script has a different default location. Use the matching `--arch`:

```
python indoor_eval.py --arch baseline --weights mobilenet_indoor.pth.tar
python indoor_eval.py --arch align    --weights mobilenet_indoor_align.pth.tar

python outdoor_eval.py --arch baseline --weights mobilenet_outdoor.pth.tar
python outdoor_eval.py --arch align    --weights mobilenet_outdoor_align.pth.tar

python cityscapes_eval.py --arch baseline --weights cityscapes_mobilenet.pth.tar
python cityscapes_eval.py --arch align    --weights cityscapes_mobilenet_align.pth.tar
```

The first line printed should say `loaded ... into the baseline head` (or `align head`). If it names the wrong head, the wrong file was loaded.

If a plot window opens, close it so the script can finish.

Take final numbers from these eval scripts, not from the training logs. The accuracy printed during training is only used to pick the best checkpoint.

## 6. What to send back

For every dataset, head and seed:

1. The full eval output (per-class precision, recall, F1, and the overall "All" F1).
2. The `.csv` training-history file.
3. The size of each `.pth.tar` file in MB.
4. Your GPU model and the total training time (printed as `Time taken` at the end of each run).

Paper reference points (baseline, "All" F1): Indoor 0.935, Outdoor 0.868, Cityscapes 0.820. Your baseline runs should land near these. If they are far off, something is wrong with the data folders, so send the output before continuing.

## Known caveats

- The Cityscapes best checkpoint is picked by training-set accuracy, not validation accuracy. This is the same for both heads, so the comparison stays fair.
- Align's expected gain is small on Indoor and Outdoor. Cityscapes People is where the biggest difference, if any, should appear.
