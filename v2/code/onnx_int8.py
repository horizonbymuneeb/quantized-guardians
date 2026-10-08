"""INT8 dynamic quantization via ONNX Runtime (the common deployment path). Usage: onnx_int8.py MODEL_KEY"""
import sys, os, time, numpy as np, pandas as pd
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import onnxruntime as ort
MODELS = {"protectai": "protectai/deberta-v3-base-prompt-injection-v2", "deepset": "deepset/deberta-v3-base-injection",
          "fmops": "fmops/distilbert-prompt-injection", "piguard": "leolee99/PIGuard"}
key = sys.argv[1]; mid = MODELS[key]
W = f"/workspace/work/qpaper/onnx/{key}"; OUT = f"/workspace/work/qpaper/scores/{key}__int8onnx.npy"
CK = OUT + ".part.npz"
if os.path.exists(OUT): sys.exit(0)
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
qpath = os.path.join(W, "q")
if not os.path.exists(os.path.join(qpath, "model_quantized.onnx")):
    m = ORTModelForSequenceClassification.from_pretrained(mid, export=True)
    m.save_pretrained(W)
    qz = ORTQuantizer.from_pretrained(W)
    qz.quantize(save_dir=qpath, quantization_config=AutoQuantizationConfig.avx2(is_static=False, per_channel=False))
    print("quantized", flush=True)
so = ort.SessionOptions(); so.intra_op_num_threads = 2
sess = ort.InferenceSession(os.path.join(qpath, "model_quantized.onnx"), so, providers=["CPUExecutionProvider"])
names = [i.name for i in sess.get_inputs()]
df = pd.read_parquet("/workspace/work/qpaper/data/evalset.parquet"); texts = df.text.tolist()
order = np.argsort([len(t) for t in texts]); probs = np.zeros(len(texts), np.float32); i = 0
if os.path.exists(CK): z = np.load(CK); probs = z["p"]; i = int(z["i"])
t0 = time.time(); BUDGET = float(os.environ.get("BUDGET", 480))
while i < len(order):
    if time.time() - t0 > BUDGET:
        np.savez(CK, p=probs, i=i); print("partial", key, i, flush=True); sys.exit(3)
    bs = 16 if len(texts[order[min(i + 15, len(order) - 1)]]) < 1000 else 4
    idx = order[i:i + bs]
    enc = tok([texts[j] for j in idx], truncation=True, max_length=512, padding=True, return_tensors="np")
    lg = sess.run(None, {n: enc[n].astype(np.int64) for n in names if n in enc})[0]
    e = np.exp(lg - lg.max(-1, keepdims=True)); probs[idx] = (e / e.sum(-1, keepdims=True))[:, 1]
    i += bs
np.save(OUT, probs)
if os.path.exists(CK): os.remove(CK)
print(key, "int8onnx", f"{time.time()-t0:.0f}s")
