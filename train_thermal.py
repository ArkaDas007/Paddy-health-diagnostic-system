# =============================================================================
# THERMAL PADDY LEAF DISEASE DETECTION — COMPLETE PIPELINE (SPEED-OPTIMIZED)
# Single cell: data -> preprocessing -> training -> tuning -> export -> app
#
# CHANGES vs original (goal: faster epochs, accuracy still #1 priority,
# final saved model chosen with fast inference in mind):
#   1. Mixed precision (float16) + XLA JIT compile on GPU. Final logits layer
#      is forced back to float32 so the loss stays numerically stable -
#      this is the standard safe pattern and does not cost accuracy.
#   2. Separable convolutions in blocks 2-4 of the CNN (block 1 stays a
#      regular Conv2D for low-level feature extraction). This is the
#      MobileNet/Xception trick: far fewer FLOPs per conv layer -> cheaper
#      epochs. It also frees room in the 1.15M param budget, which the
#      auto-shrink loop spends on a bigger dense head instead - so capacity
#      (and therefore accuracy potential) is preserved or improved.
#   3. Adaptive batch size (32 on GPU, 16 on CPU) - fewer steps/epoch on GPU.
#   4. verbose=2 instead of verbose=1 - Colab's live progress bar redraw has
#      real overhead over 80 epochs x 5 folds; this removes it.
#   5. Slightly shorter EarlyStopping/ReduceLROnPlateau patience - cuts idle
#      epochs after the model has plateaued. restore_best_weights=True still
#      guarantees you never lose the best epoch found.
#   6. Every fold model is now benchmarked for inference latency. The model
#      that gets saved as "best" is still chosen by accuracy/F1/AUC FIRST;
#      only among folds within 1% of the top F1 does the fastest one win.
#      So speed never overrides accuracy - it only breaks near-ties.
# =============================================================================

# ----------------------------- CONFIG ----------------------------------------
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent
DRIVE_DATASET = str(BASE_DIR / "data")
RUN_DIR       = str(BASE_DIR / "thermal_run")
APP_DIR       = str(BASE_DIR / "streamlit_app")

CLASSES   = ["BLB", "Blast", "healthy", "hispa", "leaf folder", "leaf spot"]
IMG_EXT   = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
IMG_SIZE  = 224           # left untouched - shrinking this trades accuracy for speed
N_FOLDS   = 5             # left untouched - fewer folds saves total time but
                          # weakens the accuracy estimate; per-epoch speed is
                          # handled instead (see changes above)
EPOCHS    = 80
SEED      = 42

# ----------------------------- SETUP ------------------------------------------
import os, sys, json, time, shutil, subprocess, re, importlib
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):  # no-op fallback if tqdm is missing
        if desc: print(desc)
        return iterable

# Reproducibility
np.random.seed(SEED); tf.random.set_seed(SEED)
DEVICE = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"[setup] TensorFlow {tf.__version__} | device: {DEVICE}")

# --- Speed: XLA JIT only (GPU). Mixed precision (float16) is intentionally
#     NOT used: with this dataset's severe class imbalance (~85/15), focal
#     loss already produces small gradients for the well-classified majority
#     class, and float16 underflows those small values toward zero. That
#     froze the conv filters near their random initial weights and made the
#     network collapse to predicting the majority class every time (visible
#     as val_auc stuck at exactly 0.5000 and recall pinned at 1.0). Given
#     this dataset is small (~500 train images/fold) epochs are already fast
#     without mixed precision - XLA compilation and separable convs supply
#     the speedup with none of that numerical risk.
USE_MIXED_PRECISION = False
USE_XLA = (DEVICE == "GPU")
if USE_XLA:
    tf.config.optimizer.set_jit(True)
    print("[setup] XLA JIT enabled for faster training (mixed precision left OFF - see comment above)")
else:
    print("[setup] Running on CPU - XLA skipped")

# --- Speed: bigger batches only make sense with a GPU behind them ---
BATCH = 32 if DEVICE == "GPU" else 16
print(f"[setup] BATCH = {BATCH}")

# Mount Drive
#if not os.path.isdir("/content/drive/MyDrive"):
#    from google.colab import drive
#    drive.mount("/content/drive")

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(APP_DIR, exist_ok=True)
print(f"[setup] RUN_DIR = {RUN_DIR}")

# =============================================================================
# PHASE 1 — DATA AUDIT
# =============================================================================
print("\n[phase 1] Data audit")

def find_data_dir(root, classes, max_depth=4):
    root = Path(root)
    for cur, _, _ in os.walk(root):
        p = Path(cur)
        if len(p.relative_to(root).parts) > max_depth: continue
        if len([c for c in classes if (p / c).is_dir()]) >= 4:
            return p
    return None

DATA_DIR = find_data_dir(DRIVE_DATASET, CLASSES)
assert DATA_DIR is not None, f"No class folders in {DRIVE_DATASET}"
print(f"[phase 1] DATA_DIR (Drive) = {DATA_DIR}")

LOCAL_DATA_DIR = BASE_DIR / "local_paddy_dataset"
if not LOCAL_DATA_DIR.exists():
    print(f"[phase 1] Copying dataset to local disk (one-time; makes every "
          f"read below much faster) -> {LOCAL_DATA_DIR}")
    for cls in CLASSES:
        src = DATA_DIR / cls
        if not src.is_dir(): continue
        dst = LOCAL_DATA_DIR / cls
        dst.mkdir(parents=True, exist_ok=True)
        files = [p for p in src.glob("*") if p.suffix.lower() in IMG_EXT]
        for p in tqdm(files, desc=f"[phase 1] copying {cls}"):
            shutil.copy2(p, dst / p.name)
else:
    print(f"[phase 1] Local copy already exists at {LOCAL_DATA_DIR}, reusing it")
DATA_DIR = LOCAL_DATA_DIR
print(f"[phase 1] DATA_DIR (local, in use from here on) = {DATA_DIR}")

rows = []
for cls in CLASSES:
    d = DATA_DIR / cls
    if not d.is_dir(): continue
    for p in d.glob("*"):
        if p.suffix.lower() in IMG_EXT:
            rows.append({"path": str(p), "label": cls})
df = pd.DataFrame(rows)

print("[phase 1] class balance:")
print(df["label"].value_counts().to_string())
print(f"[phase 1] total = {len(df)}")

# Perceptual hash grouping (leakage guard)
def phash_leaf(path, hash_size=8):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked = cv2.bitwise_and(gray, gray, mask=th)
    small = cv2.resize(masked, (32, 32))
    dct = cv2.dct(np.float32(small))
    low = dct[:hash_size, :hash_size]
    return (low > low.mean()).flatten()

hashes, paths = [], []
for p in tqdm(df["path"], desc="[phase 1] hashing (leakage check)"):
    h = phash_leaf(p)
    if h is not None:
        hashes.append(h); paths.append(p)
hashes = np.array(hashes)

group_id = -np.ones(len(hashes), dtype=int)
for i in range(len(hashes)):
    if group_id[i] != -1: continue
    gid = group_id.max() + 1
    close = np.where(np.sum(hashes != hashes[i], axis=1) <= 4)[0]
    for j in close:
        if group_id[j] == -1: group_id[j] = gid

df = df.set_index("path").loc[paths].reset_index()
df["group_id"] = group_id
print(f"[phase 1] unique groups = {df['group_id'].nunique()} / {len(df)}")

# =============================================================================
# PHASE 2 — PREPROCESSING
# =============================================================================
print("\n[phase 2] Preprocessing")

def segment_leaf(gray_u8):
    blur = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        th = (labels == largest).astype(np.uint8) * 255
    return th

def preprocess_thermal(path, size=(IMG_SIZE, IMG_SIZE), debug=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(den)
    mask = segment_leaf(enh)
    masked = cv2.bitwise_and(enh, enh, mask=mask)
    resized = cv2.resize(masked, size, interpolation=cv2.INTER_AREA)
    if debug:
        return gray, den, enh, mask, resized
    return resized.astype(np.float32) / 255.0

# Sanity grid
samp = df.sample(min(6, len(df)), random_state=1)["path"].tolist()
fig, axes = plt.subplots(len(samp), 5, figsize=(15, 3*len(samp)))
if len(samp) == 1: axes = axes[None, :]
for i, p in enumerate(samp):
    g, d, e, m, r = preprocess_thermal(p, debug=True)
    for j, (t, im) in enumerate(zip(
        ["raw","denoised","CLAHE","mask","final"], [g,d,e,m,r])):
        axes[i, j].imshow(im, cmap="gray")
        axes[i, j].set_title(f"{Path(p).parent.name} - {t}", fontsize=8)
        axes[i, j].axis("off")
plt.tight_layout()
plt.savefig(f"{RUN_DIR}/preprocess_grid.png", dpi=100)
plt.close()
print("[phase 2] sanity grid saved")

# =============================================================================
# PHASE 3 — PRECOMPUTE + SPLITS
# =============================================================================
print("\n[phase 3] Precomputing arrays")

X = np.zeros((len(df), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
y_multi = np.zeros(len(df), dtype=np.int32)
keep = np.ones(len(df), dtype=bool)
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

for i, (p, lbl) in enumerate(tqdm(list(zip(df["path"], df["label"])),
                                   desc="[phase 3] preprocessing images")):
    arr = preprocess_thermal(p)
    if arr is None: keep[i] = False; continue
    X[i] = arr[..., None]
    y_multi[i] = class_to_idx[lbl]

X, y_multi = X[keep], y_multi[keep]
groups = df["group_id"].values[keep]
y = (y_multi != class_to_idx["healthy"]).astype(np.int32)

print(f"[phase 3] X={X.shape}  healthy={int((y==0).sum())}  diseased={int((y==1).sum())}")

for name, arr in [("X",X),("y_bin",y),("y_multi",y_multi),("groups",groups)]:
    np.save(f"{RUN_DIR}/{name}.npy", arr)

skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(X, y, groups=groups))
for i, (tr, va) in enumerate(FOLDS):
    print(f"[phase 3] fold {i+1}: train={len(tr)}  val={len(va)}  "
          f"pos_tr={y[tr].mean():.2f}  pos_va={y[va].mean():.2f}")

AUTOTUNE = tf.data.AUTOTUNE
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
], name="augment")

# =============================================================================
# PHASE 4 — MODEL
# =============================================================================
print("\n[phase 4] Building model")

@register_keras_serializable(package="custom")
def focal_loss(gamma=2.0, alpha=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)  # safe even under mixed precision
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.-eps)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.-y_pred)
        at = tf.where(tf.equal(y_true, 1.0), alpha, 1.-alpha)
        return -tf.reduce_mean(at * tf.pow(1.-pt, gamma) * tf.math.log(pt))
    return loss

def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), l2=1e-4,
                     d1=512, d2=256, output_bias=None, name="custom_cnn"):
    reg = tf.keras.regularizers.l2(l2)
    inp = tf.keras.Input(input_shape)

    def block(x, f, separable):
        # Speed: SeparableConv2D has far fewer FLOPs/params than a regular
        # Conv2D for the same filter count. Block 1 stays a regular conv
        # (best at extracting raw low-level edge/thermal-gradient features);
        # blocks 2-4 go separable, which is where most of the compute lives.
        # NOTE: Keras 3's SeparableConv2D splits regularization into
        # depthwise_regularizer / pointwise_regularizer instead of a single
        # kernel_regularizer (it's really two ops: depthwise + 1x1 pointwise).
        if separable:
            x = tf.keras.layers.SeparableConv2D(
                f, 3, padding="same",
                depthwise_regularizer=reg, pointwise_regularizer=reg)(x)
        else:
            x = tf.keras.layers.Conv2D(
                f, 3, padding="same", kernel_regularizer=reg)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return tf.keras.layers.MaxPool2D(2)(x)

    x = block(inp, 32,  separable=False)
    x = block(x,   64,  separable=True)
    x = block(x,  128,  separable=True)
    x = block(x,  256,  separable=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(d1, kernel_regularizer=reg)(x)
    x = tf.keras.layers.ReLU()(x); x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(d2, kernel_regularizer=reg)(x)
    x = tf.keras.layers.ReLU()(x); x = tf.keras.layers.Dropout(0.3)(x)
    # Force float32 on the logits layer even under a mixed_float16 policy -
    # keeps the loss/log computations numerically stable. This costs
    # nothing in accuracy; it's the standard mixed-precision pattern.
    out = tf.keras.layers.Dense(1, bias_initializer=output_bias,
                                name="logits", dtype="float32")(x)
    return tf.keras.Model(inp, out, name=name)

# Auto-shrink to <=1.15M. Because separable convs free up parameters versus
# the original all-Conv2D design, this loop can now afford a same-size or
# larger dense head within the same budget -> more capacity, not less.
TARGET = 1_150_000
CNN_KWARGS = None
for d1, d2 in [(512,256),(448,224),(384,192),(320,160),(256,128),(192,96)]:
    m = build_custom_cnn(d1=d1, d2=d2)
    p = m.count_params()
    if p <= TARGET:
        CNN_KWARGS = dict(d1=d1, d2=d2); break
print(f"[phase 4] chosen {CNN_KWARGS}  params = {m.count_params():,}")

def make_output_bias(y_train):
    p = float(np.mean(y_train)); p = min(max(p, 1e-3), 1-1e-3)
    return tf.keras.initializers.Constant(np.log(p/(1-p)))

def bench_latency_ms(model, img_size=IMG_SIZE, channels=1, n=50):
    """Quick single-image inference latency benchmark (warmed up)."""
    x = tf.random.normal((1, img_size, img_size, channels))
    for _ in range(5): model(x, training=False)
    t0 = time.perf_counter()
    for _ in range(n): model(x, training=False)
    return (time.perf_counter() - t0) / n * 1000

# =============================================================================
# PHASE 5 — TRAINING (5-fold)
# =============================================================================
print("\n[phase 5] Training")

fold_metrics, fold_histories, fold_holdouts = [], {}, {}

for fold, (tr_idx, va_idx) in enumerate(FOLDS):
    print(f"\n----- fold {fold+1}/{N_FOLDS} -----")
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
                .shuffle(2048).batch(BATCH)
                .map(lambda x, y: (augment(x, training=True), y),
                     num_parallel_calls=AUTOTUNE, deterministic=False)
                .prefetch(AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((Xva, yva))
              .batch(BATCH).prefetch(AUTOTUNE))

    # Class-frequency-aware alpha: with alpha fixed at 0.5, both classes were
    # weighted equally in the loss even though diseased outnumbers healthy
    # ~6:1 in this dataset - that gave "healthy" too little gradient signal
    # to learn from. Setting alpha = 1 - (fraction diseased) upweights the
    # minority (healthy) class's loss contribution in proportion to how rare
    # it is, which is the standard fix for focal loss under imbalance.
    pos_rate = float(np.mean(ytr))
    fold_alpha = float(np.clip(1.0 - pos_rate, 0.05, 0.95))
    print(f"[fold {fold+1}] pos_rate(diseased)={pos_rate:.3f}  focal_alpha={fold_alpha:.3f}")

    model = build_custom_cnn(**CNN_KWARGS, output_bias=make_output_bias(ytr))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4, clipnorm=1.0),  # clipnorm:
            # guards against occasional large/unstable gradient steps: cheap
            # insurance, does not limit normal-sized gradients or slow training
        loss=focal_loss(gamma=2.0, alpha=fold_alpha),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        jit_compile=USE_XLA,
    )

    ckpt = f"{RUN_DIR}/model_fold{fold+1}.keras"
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                         patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min",
                                             factor=0.5, patience=5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", mode="min",
                                           save_best_only=True, verbose=0),
    ]

    t_start = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                     callbacks=cbs, verbose=2)  # verbose=2: no per-epoch progress
                                                 # bar redraw overhead
    train_time_s = time.perf_counter() - t_start
    n_epochs_ran = len(hist.history.get("loss", []))
    print(f"[fold {fold+1}] trained {n_epochs_ran} epochs in {train_time_s:.1f}s "
          f"({train_time_s / max(n_epochs_ran,1):.2f}s/epoch)")

    # TTA evaluation (horizontal + vertical flip, matching training augmentation)
    logits   = model.predict(val_ds, verbose=0).ravel()
    logits_h = model.predict(
        tf.data.Dataset.from_tensor_slices((Xva[:, :, ::-1, :], yva)).batch(BATCH),
        verbose=0).ravel()
    logits_v = model.predict(
        tf.data.Dataset.from_tensor_slices((Xva[:, ::-1, :, :], yva)).batch(BATCH),
        verbose=0).ravel()
    prob = 1.0/(1.0 + np.exp(-(logits + logits_h + logits_v) / 3.0))
    pred = (prob >= 0.5).astype(int)

    latency_ms = bench_latency_ms(model)

    m = {"fold": fold+1,
         "acc":  float(accuracy_score(yva, pred)),
         "prec": float(precision_score(yva, pred, zero_division=0)),
         "rec":  float(recall_score(yva, pred, zero_division=0)),
         "f1":   float(f1_score(yva, pred, zero_division=0)),
         "auc":  float(roc_auc_score(yva, prob)),
         "train_time_s": float(train_time_s),
         "epochs_ran": int(n_epochs_ran),
         "latency_ms": float(latency_ms),
         "focal_alpha": float(fold_alpha)}
    fold_metrics.append(m)
    fold_histories[fold] = hist.history
    fold_holdouts[fold] = (va_idx, yva, prob)
    print(f"[fold {fold+1}] {m}")
    print(f"[fold {fold+1}] healthy mean P = {prob[yva==0].mean():.3f} | "
          f"diseased mean P = {prob[yva==1].mean():.3f}")

    with open(f"{RUN_DIR}/hist_fold{fold+1}.json", "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()}, f)

fm = pd.DataFrame(fold_metrics)
summary = fm.drop(columns="fold").agg(["mean","std"]).T
print("\n[phase 5] K-fold summary:")
print(summary)
fm.to_csv(f"{RUN_DIR}/fold_metrics.csv", index=False)
summary.to_csv(f"{RUN_DIR}/fold_summary.csv")

# =============================================================================
# PHASE 6 — AGGREGATED PREDICTIONS + DIAGNOSTIC
# =============================================================================
print("\n[phase 6] Aggregating predictions")

all_true, all_pred, all_prob = [], [], []
for fold, (va_idx, yva, _) in fold_holdouts.items():
    m = tf.keras.models.load_model(f"{RUN_DIR}/model_fold{fold+1}.keras",
                                   compile=False)
    logits   = m.predict(X[va_idx], verbose=0).ravel()
    logits_h = m.predict(X[va_idx][:, :, ::-1, :], verbose=0).ravel()
    logits_v = m.predict(X[va_idx][:, ::-1, :, :], verbose=0).ravel()
    prob = 1.0/(1.0 + np.exp(-(logits + logits_h + logits_v) / 3.0))
    all_true.extend(yva); all_pred.extend((prob>=0.5).astype(int)); all_prob.extend(prob)

all_true = np.array(all_true); all_pred = np.array(all_pred); all_prob = np.array(all_prob)

hp = all_prob[all_true==0]; dp = all_prob[all_true==1]
print(f"[phase 6] Healthy  mean P = {hp.mean():.3f}  (n={len(hp)})")
print(f"[phase 6] Diseased mean P = {dp.mean():.3f}  (n={len(dp)})")

plt.figure(figsize=(10, 4))
plt.hist(hp, bins=30, alpha=.6, color="green", label="true Healthy")
plt.hist(dp, bins=30, alpha=.6, color="red",   label="true Diseased")
plt.axvline(0.5, color="black", ls="--", label="0.5")
plt.xlabel("P(diseased)"); plt.legend()
plt.title("Distribution of P(diseased) per true class")
plt.tight_layout()
plt.savefig(f"{RUN_DIR}/prob_distribution.png", dpi=120); plt.close()

# =============================================================================
# PHASE 7 — THRESHOLD TUNING
# =============================================================================
print("\n[phase 7] Threshold tuning")

fpr, tpr, thr_roc = roc_curve(all_true, all_prob)
J = tpr - fpr
CHOSEN_THR = float(thr_roc[int(np.argmax(J))])
print(f"[phase 7] Youden's J threshold = {CHOSEN_THR:.4f}")

pred_tuned = (all_prob >= CHOSEN_THR).astype(int)
final_metrics = {
    "acc":  float(accuracy_score(all_true, pred_tuned)),
    "prec": float(precision_score(all_true, pred_tuned, zero_division=0)),
    "rec":  float(recall_score(all_true, pred_tuned, zero_division=0)),
    "f1":   float(f1_score(all_true, pred_tuned, zero_division=0)),
    "auc":  float(roc_auc_score(all_true, all_prob)),
}
print("[phase 7] tuned metrics:", final_metrics)

cm = confusion_matrix(all_true, pred_tuned)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy","Diseased"],
            yticklabels=["Healthy","Diseased"])
plt.title(f"CM @ threshold={CHOSEN_THR:.3f}")
plt.tight_layout()
plt.savefig(f"{RUN_DIR}/cm_tuned.png", dpi=120); plt.close()

# =============================================================================
# PHASE 8 — BEST-MODEL SELECTION (accuracy first, speed as tiebreaker)
# =============================================================================
print("\n[phase 8] Selecting best model")

# Rule: rank folds by AUC FIRST, not F1/accuracy. Under this dataset's ~85/15
# imbalance, a model that just predicts "diseased" for every image already
# scores ~85% accuracy and a high F1 (recall pinned at 1.0) despite having
# zero real diagnostic skill - that is exactly what happened when mixed
# precision broke training above (fold 4 had the *top* F1 with AUC=0.10,
# worse than a coin flip). AUC measures true class-separation ability and
# is not fooled by a constant predictor, so it is the real "accuracy first"
# metric here. Folds within 1% of the top AUC are treated as equally good;
# F1 breaks ties within that tier, and latency only breaks ties after that -
# so speed still never overrides genuine accuracy, it just settles ties.
AUC_TOLERANCE = 0.01
top_auc = fm["auc"].max()
tier = fm[fm["auc"] >= top_auc - AUC_TOLERANCE].copy()
tier = tier.sort_values(["f1", "latency_ms"], ascending=[False, True])
best_row = tier.iloc[0]
best_fold = int(best_row["fold"])

print(f"[phase 8] top AUC = {top_auc:.4f} | candidates within {AUC_TOLERANCE} of top:")
print(tier[["fold","auc","f1","latency_ms","train_time_s"]].to_string(index=False))
print(f"[phase 8] selected fold {best_fold}  "
      f"(auc={best_row['auc']:.4f}, f1={best_row['f1']:.4f}, "
      f"latency={best_row['latency_ms']:.2f} ms)")

if top_auc < 0.65:
    print(f"[phase 8] WARNING: best AUC is only {top_auc:.3f}. This usually "
          f"means the model is closer to guessing than actually discriminating "
          f"healthy vs diseased. If you see this after the fixes in this "
          f"version, the likely next step is more healthy-class images "
          f"(currently only 93 vs 543 diseased) rather than further tuning.")

best_model = tf.keras.models.load_model(f"{RUN_DIR}/model_fold{best_fold}.keras",
                                        compile=False)

# =============================================================================
# PHASE 9 — GRAD-CAM (best fold)
# =============================================================================
print("\n[phase 9] Grad-CAM")

def gradcam_on_logit(model, img_2d):
    layer_name = None
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)
    for l in reversed(model.layers):
        if isinstance(l, conv_types):
            layer_name = l.name; break
    gm = tf.keras.Model([model.inputs],
                        [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv, preds = gm(tf.expand_dims(img_2d, 0))
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv)
    # Defensive float32 cast: under any future mixed-precision policy the
    # conv/grad tensors would be float16, and cv2.resize() (called on this
    # output downstream) crashes on float16 input rather than erroring
    # clearly. Casting here makes Grad-CAM safe regardless of policy.
    conv = tf.cast(conv, tf.float32)
    grads = tf.cast(grads, tf.float32)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heat = tf.nn.relu(tf.reduce_sum(conv[0] * pooled, axis=-1)).numpy()
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    hmax = heat.max()
    return heat / hmax if hmax > 1e-8 else heat

idxs = np.random.RandomState(0).choice(len(X), min(6, len(X)), replace=False)
fig, axes = plt.subplots(2, len(idxs), figsize=(20, 7))
if len(idxs) == 1: axes = axes[:, None]
for j, i in enumerate(idxs):
    img = X[i]
    heat = cv2.resize(gradcam_on_logit(best_model, img),
                      (img.shape[1], img.shape[0]))
    overlay = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
    axes[0, j].imshow(img.squeeze(), cmap="gray")
    axes[0, j].set_title(f"y={y[i]}", fontsize=9); axes[0, j].axis("off")
    axes[1, j].imshow(img.squeeze(), cmap="gray")
    axes[1, j].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), alpha=.5)
    axes[1, j].axis("off")
plt.tight_layout()
plt.savefig(f"{RUN_DIR}/gradcam.png", dpi=120); plt.close()

# =============================================================================
# PHASE 10 — BENCHMARK + EXPORT
# =============================================================================
print("\n[phase 10] Benchmark + export")

lat_ms = bench_latency_ms(best_model)
print(f"[phase 10] params={best_model.count_params():,}  latency={lat_ms:.1f} ms")

# Copy best model
shutil.copy(f"{RUN_DIR}/model_fold{best_fold}.keras",
            f"{RUN_DIR}/model_best.keras")

# INT8 TFLite
try:
    conv = tf.lite.TFLiteConverter.from_keras_model(best_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    def rep_gen():
        for i in range(min(200, len(X))):
            yield [X[i:i+1].astype(np.float32)]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.uint8
    conv.inference_output_type = tf.uint8
    tfl = conv.convert()
    with open(f"{RUN_DIR}/model_int8.tflite","wb") as f: f.write(tfl)
    print(f"[phase 10] INT8 size = {os.path.getsize(f'{RUN_DIR}/model_int8.tflite')/1024**2:.2f} MB")
except Exception as e:
    print(f"[phase 10] INT8 skipped: {e}")

# Save meta
meta = {
    "version": "v4-speed",
    "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
    "input_shape": [IMG_SIZE, IMG_SIZE, 1],
    "image_size": [IMG_SIZE, IMG_SIZE],
    "normalization": {"scale": 1.0/255.0},
    "segmentation": {"method": "otsu_plus_morphology", "blur_ksize": 5,
                     "morph_close_iters": 2, "morph_open_iters": 1,
                     "keep_largest_cc": True},
    "denoise": {"method": "fastNlMeansDenoising", "h": 10},
    "clahe": {"clipLimit": 2.0, "tileGridSize": [8, 8]},
    "decision_threshold": CHOSEN_THR,
    "output_activation": "logits",
    "label_map": {"0": "Healthy", "1": "Diseased"},
    "class_to_idx": class_to_idx,
    "cnn_kwargs": CNN_KWARGS,
    "architecture": "separable_conv_blocks_2to4",
    "mixed_precision": USE_MIXED_PRECISION,
    "xla": USE_XLA,
    "best_fold": best_fold,
    "params": int(best_model.count_params()),
    "latency_ms": float(lat_ms),
    "final_metrics": final_metrics,
    "model_selection_rule": (
        f"Highest AUC wins first (robust to class-imbalance-inflated "
        f"accuracy/F1/recall); folds within {AUC_TOLERANCE} of top AUC are "
        "treated as equally accurate, and among those the highest F1 is kept, "
        "with lowest latency as the final tiebreaker."
    ),
    "fold_metrics": fold_metrics,
}
with open(f"{RUN_DIR}/model_meta.json","w") as f:
    json.dump(meta, f, indent=2)
print("[phase 10] meta saved")

# =============================================================================
# PHASE 11 — PREDICT.PY
# =============================================================================
print("\n[phase 11] Writing predict.py")

PREDICT_CODE = '''
import os, json
import numpy as np
import cv2
import tensorflow as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "model_meta.json")) as f:
    META = json.load(f)
_model = tf.keras.models.load_model(os.path.join(_HERE, "model_best.keras"),
                                    compile=False)

_den_h  = META["denoise"]["h"]
_clahe  = cv2.createCLAHE(clipLimit=META["clahe"]["clipLimit"],
                          tileGridSize=tuple(META["clahe"]["tileGridSize"]))
_size   = tuple(META["image_size"])
_blur_k = META["segmentation"]["blur_ksize"]


def _segment(gray_u8):
    blur = cv2.GaussianBlur(gray_u8, (_blur_k, _blur_k), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k,
                          iterations=META["segmentation"]["morph_close_iters"])
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k,
                          iterations=META["segmentation"]["morph_open_iters"])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if n > 1 and META["segmentation"]["keep_largest_cc"]:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        th = (labels == largest).astype(np.uint8) * 255
    return th


def preprocess_external(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(gray, h=_den_h)
    enh = _clahe.apply(den)
    mask = _segment(enh)
    masked = cv2.bitwise_and(enh, enh, mask=mask)
    resized = cv2.resize(masked, _size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0, resized


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def _gradcam(img_2d):
    m = _model
    layer_name = None
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)
    for l in reversed(m.layers):
        if isinstance(l, conv_types): layer_name = l.name; break
    gm = tf.keras.Model([m.inputs], [m.get_layer(layer_name).output, m.output])
    with tf.GradientTape() as tape:
        conv, preds = gm(tf.expand_dims(img_2d, 0))
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv)
    # Defensive float32 cast + NaN guard - see training script for why.
    conv = tf.cast(conv, tf.float32)
    grads = tf.cast(grads, tf.float32)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heat = tf.nn.relu(tf.reduce_sum(conv[0]*pooled, axis=-1)).numpy()
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    hmax = heat.max()
    return heat / hmax if hmax > 1e-8 else heat


def predict(image_path, return_gradcam=False, return_thermal_stats=False):
    arr, u8 = preprocess_external(image_path)
    logit = float(_model.predict(arr[..., None][None, ...], verbose=0).ravel()[0])
    p = _sigmoid(logit)
    thr = META["decision_threshold"]
    label = META["label_map"]["1"] if p >= thr else META["label_map"]["0"]
    out = {"label": label,
           "confidence": p if p >= thr else 1 - p,
           "prob_diseased": p, "logit": logit,
           "threshold": thr, "preprocessed_u8": u8}
    if return_gradcam:
        out["gradcam"] = _gradcam(arr[..., None])
    if return_thermal_stats:
        v = u8[u8 > 0].astype(np.float32)
        if v.size == 0:
            out["thermal_stats"] = {"mean":0.,"std":0.,"p90_minus_p10":0.,"hot_frac":0.}
        else:
            mu, sd = float(v.mean()), float(v.std())
            p10, p90 = np.percentile(v, [10, 90])
            out["thermal_stats"] = {"mean": mu, "std": sd,
                                    "p90_minus_p10": float(p90-p10),
                                    "hot_frac": float((v > (mu+2*sd)).mean())}
    return out
'''

with open(f"{RUN_DIR}/predict.py","w") as f: f.write(PREDICT_CODE)
print("[phase 11] predict.py written")

# =============================================================================
# PHASE 12 — STREAMLIT APP
# =============================================================================
print("\n[phase 12] Writing streamlit_app.py")

STREAMLIT_CODE = '''
import os, tempfile
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict import predict, META

st.set_page_config(page_title="Paddy Thermal Disease Detector",
                   page_icon="\\U0001F33E", layout="wide")

st.title("\\U0001F33E Paddy Leaf Disease Detection (Thermal)")
st.caption("Upload a thermal image of a paddy leaf. "
           "Model classifies it as **Healthy** or **Diseased**.")

with st.sidebar:
    st.header("Settings")
    show_gradcam = st.checkbox("Show Grad-CAM", value=True)
    show_thermal = st.checkbox("Show thermal stats", value=True)
    st.markdown("---")
    st.markdown(f"**Version:** `{META.get('version','?')}`")
    st.markdown(f"**Threshold:** `{META['decision_threshold']:.3f}`")
    st.markdown(f"**Params:** `{META.get('params','?')}`")
    st.markdown(f"**Latency:** `{META.get('latency_ms','?'):.1f} ms`"
                if isinstance(META.get('latency_ms'), (int, float)) else "")

uploaded = st.file_uploader("Upload thermal image",
                            type=["jpg","jpeg","png","tif","tiff","bmp"])
if uploaded is None:
    st.info("Awaiting an image."); st.stop()

suffix = os.path.splitext(uploaded.name)[1] or ".png"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read()); tmp_path = tmp.name

with st.spinner("Analyzing..."):
    res = predict(tmp_path, return_gradcam=show_gradcam,
                  return_thermal_stats=show_thermal)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Original")
    st.image(Image.open(tmp_path), use_container_width=True)
with c2:
    st.subheader("Preprocessed leaf")
    st.image(res["preprocessed_u8"], use_container_width=True, clamp=True)

st.markdown("---")
lbl, conf = res["label"], res["confidence"]*100
if lbl == "Diseased":
    st.error(f"### \\u26A0\\uFE0F {lbl}  —  Confidence: **{conf:.1f}%**")
else:
    st.success(f"### \\u2705 {lbl}  —  Confidence: **{conf:.1f}%**")
st.progress(min(max(res["prob_diseased"], 0.), 1.))
st.caption(f"P(diseased) = {res['prob_diseased']:.4f} | "
           f"threshold = {res['threshold']:.3f}")

if show_gradcam and "gradcam" in res:
    st.markdown("### Grad-CAM")
    heat = res["gradcam"]
    heat_r = cv2.resize(heat, (res["preprocessed_u8"].shape[1],
                               res["preprocessed_u8"].shape[0]))
    overlay = cv2.applyColorMap((heat_r*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(res["preprocessed_u8"], cmap="gray"); axes[0].set_title("Leaf")
    axes[1].imshow(heat_r, cmap="jet"); axes[1].set_title("Heatmap")
    axes[2].imshow(res["preprocessed_u8"], cmap="gray")
    axes[2].imshow(overlay, alpha=0.5); axes[2].set_title("Overlay")
    for ax in axes: ax.axis("off")
    st.pyplot(fig, clear_figure=True)

if show_thermal and "thermal_stats" in res:
    st.markdown("### Thermal stats (not validated CWSI)")
    st.json({k: round(v, 4) for k, v in res["thermal_stats"].items()})

st.markdown("---")
st.caption("\\u26A0\\uFE0F Research prototype. Not for agronomic decisions.")
'''

with open(f"{RUN_DIR}/streamlit_app.py","w") as f: f.write(STREAMLIT_CODE)
print("[phase 12] streamlit_app.py written")

# =============================================================================
# PHASE 13 — DEPLOY
# =============================================================================
print("\n[phase 13] Deploying Streamlit")

# Fresh APP_DIR
if os.path.exists(APP_DIR): shutil.rmtree(APP_DIR)
os.makedirs(APP_DIR, exist_ok=True)
for f_ in ["predict.py", "streamlit_app.py", "model_best.keras", "model_meta.json"]:
    shutil.copy(f"{RUN_DIR}/{f_}", f"{APP_DIR}/{f_}")
print(f"[phase 13] app folder ready: {sorted(os.listdir(APP_DIR))}")

# Install cloudflared if missing
if not os.path.exists("/usr/local/bin/cloudflared"):
    subprocess.run(
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/"
        "download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared "
        "&& chmod +x /usr/local/bin/cloudflared",
        shell=True, check=True)

# Install streamlit if missing - Colab does not ship it preinstalled, which
# is what the previous "FileNotFoundError: streamlit" crash was.
try:
    import streamlit  # noqa: F401
    print("[phase 13] streamlit already installed")
except ImportError:
    print("[phase 13] streamlit not found - installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "streamlit"],
                   check=True)

# Kill stale processes
subprocess.run("pkill -f streamlit || true", shell=True)
subprocess.run("pkill -f cloudflared || true", shell=True)
time.sleep(2)

# Start Streamlit - launched as `python -m streamlit` (not the bare
# "streamlit" command) so it works even if the console-script entry point
# isn't on PATH; this always resolves as long as the package is importable
# by this same interpreter, which the install step above guarantees.
subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", f"{APP_DIR}/streamlit_app.py",
     "--server.port", "8501", "--server.headless", "true",
     "--browser.gatherUsageStats", "false"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(8)
print("[phase 13] Streamlit started")

# Cloudflared tunnel
tunnel = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8501",
     "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

public_url = None
start = time.time()
while time.time() - start < 60:
    line = tunnel.stdout.readline()
    if not line: continue
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        public_url = m.group(0); break

print("\n" + "="*70)
if public_url:
    print(f"[OK] STREAMLIT APP LIVE AT: {public_url}")
else:
    print("[!] Could not auto-detect URL. Check cloudflared output above.")
print("="*70)

# =============================================================================
# PHASE 14 — SANITY TEST
# =============================================================================
print("\n[phase 14] Per-class sanity test")

for k in list(sys.modules.keys()):
    if k == "predict" or k.startswith("predict."):
        del sys.modules[k]
sys.path = [APP_DIR] + [p for p in sys.path if p != APP_DIR]
importlib.invalidate_caches()
import predict as P
importlib.reload(P)

print(f"[phase 14] loaded model {P.META.get('version')} | "
      f"thr = {P.META['decision_threshold']:.4f} | "
      f"latency = {P.META.get('latency_ms', 0):.1f} ms\n")

for cls in CLASSES:
    sub = df[df["label"] == cls]
    if len(sub) == 0: continue
    sample = sub.sample(min(3, len(sub)), random_state=0)
    results = []
    for _, row in sample.iterrows():
        r = P.predict(row["path"])
        results.append(f"{r['label']}(P={r['prob_diseased']:.2f})")
    print(f"{cls:12s} -> {', '.join(results)}")

print("\n[phase 14] done. If healthy predicts Healthy, you're ready.")
print("="*70)