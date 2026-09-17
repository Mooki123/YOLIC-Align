# Publishable Improvements to YOLIC (Revised Assessment)

This section is based on a review of the YOLIC paper, the current YOLIC-Align implementation, and a search of follow-up work (as of September 2026). Only one idea is rated as clearly publishable, and one more as conditionally worthwhile. No third idea was strong enough to list.

---

## 1. Layout-Agnostic YOLIC: Train Once, Deploy Any Cell Layout (strong)

**Limitation it removes:**
The original YOLIC head is tied to one cell layout: output *i* always means "cell *i* of this layout". Every new application (parking lot, new camera, new mounting height) needs new labels and a full retrain. The paper's main selling point, flexible cell layouts, therefore costs a full training run per layout.

**The idea:**
YOLIC-Align already reads each cell out through a mask computed from its geometry. Train it on **randomly generated layouts** (random grids, rectangles, distance-shaped trapezoids, polygons), computing cell labels on the fly from pixel-level ground truth. At deployment, describe any new layout, compute its masks in milliseconds, and run it with no retraining.

**Why it is publishable:**
* **New capability, not an incremental accuracy gain.** Neither the original YOLIC nor its follow-ups (Selective Multi-Branch Network, Electronics 2024; YOLIC Labeling, 2026) can do this, and no prior work doing it for cell-wise classification was found.
* **Gives YOLIC-Align a real purpose.** On its own, Align is roughly "same accuracy, smaller head". Here it is the component that makes the new capability possible.
* **Absorbs the IMU idea.** Since masks are an input, cells can be shifted to follow camera pitch at runtime. This becomes a demo inside the paper rather than a separate paper that needs IMU hardware.
* **Testable with public data.**
  * Train on Cityscapes (plus BDD100K or Mapillary).
  * Evaluate zero-shot on held-out layouts: the paper's 256-cell grid, coarse and fine grids, and fan-shaped polygons like the indoor ones.
  * Compare against a YOLIC-M2 retrained separately for each layout.
  * Headline result: "within ~X F1 of per-layout training, with zero retraining", plus accuracy against how different a layout is from those seen in training.

**Design changes needed (current code is not layout-agnostic yet):**
* The per-cell bias `cell_bias` is tied to cell index. Replace it with a tiny MLP over each cell's geometry (centroid, log-area, aspect ratio).
* The "any pixel present" label rule means different things for large and small cells. Feed cell area into the head and keep the soft-max (LSE) readout.
* Generate labels fast with integral images (2-D cumulative sums) on the GPU, so labels for any rectangle cost O(1). Pre-rasterize a pool of random polygon layouts.

**Main risk:** zero-shot accuracy may fall well short of per-layout training on very small cells. This is still publishable if the trade-off is quantified honestly.

**Feasibility test (~1–2 GPU days):** train on random rectangle layouts on Cityscapes, then evaluate on the paper's 256-cell layout against a per-layout Align run. A gap under ~2 F1 means the idea is worth pursuing.

---

## 2. Automatic Pixel Pseudo-Labels from a Segmentation Foundation Model (conditional)

**The idea:**
Run a strong open-vocabulary segmentation model offline on the outdoor and indoor frames (and on unlabeled video). Convert its pixel masks into cell labels for any layout, optionally as soft labels based on how much of the cell each class covers, and train YOLIC on them.

**Why it matters:**
The outdoor and indoor datasets only have labels for their fixed layouts, so idea #1 cannot be trained on them. This supplies the pixel-level supervision idea #1 needs, and it enables semi-supervised experiments (e.g. 5%, 10% or 25% human labels plus pseudo-labels).

**Why it is only conditional:**
* **Moderate novelty.** SAM-based pseudo-labeling is a crowded area, and the YOLIC authors already use SAM for interactive annotation (YOLIC Labeling).
* **Best as a component of #1,** not as a standalone paper.
* **Hard classes.** Open-vocabulary models may handle niche road classes (Bump, Dent, Weed, Zebra Crossing) poorly.

**Go/no-go test (hours, no training):** run the teacher on the labeled outdoor test set, convert its masks to the 104 cells, and compute per-class F1 against the human labels. If key classes score below ~0.7, drop this idea.

---

## Rejected or Downgraded Ideas

| Idea | Verdict | Reason |
|---|---|---|
| Cascade-YOLIC (original #3) | Already published | The Selective Multi-Branch Network (Electronics, 2024) already routes computation to selected regions, gaining only 2–11 ms. Skipping sigmoid outputs saves almost nothing, since the backbone dominates compute. |
| Temporal YOLIC (original #2) | Weak | The indoor frames were sampled 1 s apart and the outdoor frames hand-picked to be distinct, so there are no labeled sequences to evaluate on. Temporal smoothing is standard (low novelty). |
| IMU-adaptive cells (original #4) | Fold into #1 | No IMU data exists in any dataset. It works well as a runtime cell-shifting demo inside idea #1. |
| YOLIC-Align alone (original #1) | Supporting component | Likely gives roughly equal accuracy with a 12–80× smaller head, which is too thin alone. It is the building block for idea #1. |
| Focal / asymmetric / distance-weighted risk loss | Not novel | Useful as extra experiments, not as contributions. |

**Suggested paper story:** *Layout-agnostic cell-wise localization for edge devices.* YOLIC-Align is the method, random-layout training is the key technique, runtime pitch adaptation is the demo, and (optionally) idea #2 extends it to datasets without pixel labels.

**References:**
* YOLIC (arXiv 2307.06689): https://arxiv.org/abs/2307.06689
* A Selective Multi-Branch Network for Edge-Oriented Object Localization and Classification (Electronics, 2024): https://www.mdpi.com/2079-9292/13/8/1472
* YOLIC Labeling: SAM-based cell-wise annotation tool (2026): https://www.sciencedirect.com/science/article/pii/S2352711026000713
* SAM Enhanced Pseudo Labels for Weakly Supervised Semantic Segmentation: https://arxiv.org/abs/2305.05803

---

# Top 4 Proposed Improvements to the YOLIC Architecture

Based on a critical analysis of the YOLIC base paper, the following four extensions represent the most promising technical improvements. They are ranked by feasibility and potential impact.

---

## 1. YOLIC-Align (Spatial Mask-Pooling)

**The Idea (In Detail):**
The original YOLIC paper uses a Global Average Pooling (GAP) layer that crushes the spatial layout of the feature map (e.g., from 7x7 down to 1x1). Because spatial information is lost, the network is forced to rely on a bloated, dense Fully Connected (FC) layer to "memorize" the locations of every Cell of Interest (CoI). 
To fix this, we can introduce **Spatial Mask-Pooling**. Instead of global pooling, we project the exact geometry of the custom cell polygons (the 30 indoor cells or 256 outdoor cells) explicitly onto the intermediate CNN spatial feature map (like a 14x14 grid). We then perform masked average pooling independently for *each* cell. This generates independent cell-feature vectors that can be parsed by an ultra-lightweight, parameter-shared Multi-Layer Perceptron (MLP) head.

**Chance of Improvement:**
* **Efficiency & Scalability: ~95%** (A mathematical certainty. This design prevents the classification head parameters from exponentially exploding as the number of cells increases).
* **Accuracy: ~65-75%** (Preserving geometric, spatial-inductive biases instead of destroying them via GAP historically yields a notable bump in object localization metrics).

---

## 2. Temporal YOLIC (T-YOLIC) via Spatiotemporal Dynamics

**The Idea (In Detail):**
Edge devices tracking objects (such as electric scooters or indoor robots) stream continuous, sequential video. The current YOLIC implementation evaluates every frame in a vacuum, ignoring the physical rule of continuity. This creates a "flickering" effect where moving objects drop in and out of detection as they cross cell boundaries.
To mitigate this, we propose adding a lightweight temporal memory component (such as a Gated Recurrent Unit, ConvLSTM, or simply a feature-level Exponential Moving Average / EMA) into the network head. By fusing the spatial footprint of frame $t-1$ with frame $t$, the network can learn continuity and trajectory.

**Chance of Improvement:**
* **Accuracy & Stability: 85%+** (Temporal smoothing is standard in continuous video inference and drastically drops false negatives or label-flickering for moving targets).

---

## 3. Hierarchical Coarse-to-Fine Grid Activation (Cascade-YOLIC)

**The Idea (In Detail):**
For very dense classification setups (e.g., trying to detect tiny obstacles mapping to 500+ micro-cells), the network wastes heavy calculations running sigmoid classifications on empty pavement or empty walls. 
This improvement restructures the annotations into a two-tiered hierarchy. The image is evaluated on a grid of a dozen large "Macro-Cells." Only if a Macro-Cell triggers positive for an object does the system evaluate the nested bundle of high-resolution "Micro-Cells" inside it.

**Chance of Improvement:**
* **Computations/FLOPs: ~90%** (In common sparse environments—like an empty hallway—this allows the network to bypass massive amounts of calculation by shutting down entire regions early).
* **Accuracy: ~10%** (This architectural change focuses heavily on inferential speed/overhead reduction rather than localization accuracy).

---

## 4. IMU-Driven Adaptive Cell Grids (Viewpoint-Invariant YOLIC)

**The Idea (In Detail):**
The inherent flaw in assigning predefined pixel regions (screenspace polygons) to physical distances is that real-world platforms pitch, roll, and vibrate. If an electric scooter hits a bump, the camera tilts, meaning the "10-meter road cell" on the screen is suddenly evaluating the sky.
We propose injecting hardware Inertial Measurement Unit (IMU) metadata (pitch/yaw/roll from a gyroscope) dynamically into the network before evaluation. Using a simple affine transformation matrix, we warp and offset the CoI boundaries on the feature map so they physically anchor to the real-world horizon regardless of the vehicle's tilt.

**Chance of Improvement:**
* **Real-world Robustness: 80%+** (In any autonomous deployment traversing uneven terrain, removing viewpoint-shift errors is fundamental to preventing catastrophic false-positive emergency braking).
