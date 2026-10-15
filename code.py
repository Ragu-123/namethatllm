import os
import sys
import subprocess
import shutil

print("=== OPERA Modular Pipeline Runner on Kaggle ===", flush=True)

REPO_URL = "https://github.com/Ragu-123/namethatllm.git"
CLONE_DIR = "/kaggle/working/namethatllm" if os.path.exists("/kaggle/working") else "./namethatllm_clone"

# Clear any stale corrupted embedding caches from previous run
for stale in ["/kaggle/working/qwen_train_emb.npz", "/kaggle/working/qwen_test_emb.npz"]:
    if os.path.exists(stale):
        try:
            os.remove(stale)
            print(f"Removed stale cache: {stale}", flush=True)
        except Exception:
            pass

# 1. Clone or update repository from GitHub
if not os.path.exists(CLONE_DIR):
    print(f"Cloning repository from {REPO_URL} into {CLONE_DIR}...", flush=True)
    res = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Git clone failed: {res.stderr}", flush=True)
    else:
        print(">> Repository successfully cloned from GitHub!", flush=True)
else:
    print(f"Repository directory {CLONE_DIR} already exists. Pulling latest from GitHub...", flush=True)
    res = subprocess.run(["git", "-C", CLONE_DIR, "pull"], capture_output=True, text=True)
    print(f"Git pull output: {res.stdout.strip()}", flush=True)

# 2. Add repo to Python path
if os.path.exists(CLONE_DIR):
    if CLONE_DIR not in sys.path:
        sys.path.insert(0, CLONE_DIR)
    print(f"Added {CLONE_DIR} to sys.path", flush=True)

# 3. Auto-detect data and model directories
data_candidates = [
    '/kaggle/input/competitions/name-that-llm',
    '/kaggle/input/name-that-llm',
    './data',
]
data_dir = next((d for d in data_candidates if os.path.exists(os.path.join(d, 'train.csv'))), None)
if data_dir is None:
    raise FileNotFoundError("Could not find train.csv in any candidate directory!")

qwen_candidates = [
    "/kaggle/input/models/qwen-lm/qwen-3/transformers/1.7b-base/1",
    "/kaggle/input/qwen-3/transformers/1.7b-base/1",
]
qwen_dir = next((p for p in qwen_candidates if os.path.exists(os.path.join(p, 'config.json'))), None)

out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."

print(f"\nConfiguration:")
print(f"  Data Directory: {data_dir}")
print(f"  Qwen 3 Directory: {qwen_dir}")
print(f"  Output Directory: {out_dir}")

# 4. Import modular OPERA pipeline and execute
from opera.pipeline import run_opera_pipeline

print("\nStarting execution of modular OPERA pipeline with verified Qwen 3 embeddings...", flush=True)
best_ba = run_opera_pipeline(
    data_dir=data_dir,
    qwen_dir=qwen_dir,
    output_dir=out_dir,
    use_neural=(qwen_dir is not None)
)
print(f"\n>> Pipeline finished successfully with Calibrated Balanced Accuracy: {best_ba:.5f}!", flush=True)
