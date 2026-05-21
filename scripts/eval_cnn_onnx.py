"""Quick ONNX test accuracy (same NCHW layout as Java)."""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.path.join(ROOT, "dataset_filtered", "test")
MODELS = os.path.join(ROOT, "models")
SIZE = 64


def classes(train_or_test, speed_only):
    out = []
    for name in sorted(os.listdir(train_or_test)):
        p = os.path.join(train_or_test, name)
        if os.path.isdir(p) and speed_only == name.startswith("Speed_limit_"):
            out.append(name)
    return out


def load_labels(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def blob(img_bgr):
    img = cv2.cvtColor(cv2.resize(img_bgr, (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    chw = np.transpose(img, (2, 0, 1))
    return chw[np.newaxis, ...]


def eval_split(speed_only):
    import onnxruntime as ort

    prefix = "speed" if speed_only else "other"
    labels = load_labels(os.path.join(MODELS, f"{prefix}_cnn_labels.txt"))
    sess = ort.InferenceSession(os.path.join(MODELS, f"{prefix}_cnn.onnx"))
    inp = sess.get_inputs()[0].name

    ok = tot = 0
    for label in labels:
        folder = os.path.join(TEST, label)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(folder, fn))
            if img is None:
                continue
            pred = labels[int(np.argmax(sess.run(None, {inp: blob(img)})[0]))]
            tot += 1
            if pred == label:
                ok += 1
    name = "vitesse" if speed_only else "autres"
    pct = 100.0 * ok / tot if tot else 0
    print(f"{name}: {ok}/{tot} = {pct:.1f}%")
    return pct


def main():
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("pip install onnxruntime")
        sys.exit(1)
    eval_split(True)
    eval_split(False)


if __name__ == "__main__":
    main()
