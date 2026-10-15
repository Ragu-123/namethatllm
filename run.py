import os
import sys

# Auto-detect data and model directories
data_candidates = [
    '/kaggle/input/competitions/name-that-llm',
    '/kaggle/input/name-that-llm',
    './data',
    '../data',
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

from opera.pipeline import run_opera_pipeline

if __name__ == "__main__":
    print(f"Data directory: {data_dir}")
    print(f"Qwen directory: {qwen_dir}")
    print(f"Output directory: {out_dir}")
    run_opera_pipeline(data_dir=data_dir, qwen_dir=qwen_dir, output_dir=out_dir, use_neural=True)
