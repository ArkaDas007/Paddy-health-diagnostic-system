
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
