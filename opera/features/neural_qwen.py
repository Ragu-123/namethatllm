import os
import gc
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def extract_qwen_features(texts, model_dir, cache_path=None, batch_size=16, max_length=512):
    """Extract verified, finite mean-pooled Layer 17 & Layer 20 hidden states from Qwen 3 (1.7B-Base)."""
    if cache_path and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            h17 = data["h17"]
            h20 = data["h20"]
            if np.isfinite(h17).all() and np.isfinite(h20).all() and len(h17) == len(texts):
                print(f">> Loaded verified cached Qwen representations from {cache_path} (shape={h17.shape})", flush=True)
                return h17, h20
            else:
                print(f">> Cache {cache_path} failed validity check (non-finite or shape mismatch). Re-extracting...", flush=True)
        except Exception as e:
            print(f">> Cache read error: {e}. Re-extracting...", flush=True)

    print(f">> Extracting Qwen 3 representations ({len(texts)} passages, batch_size={batch_size})...", flush=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        local_files_only=True,
        device_map={"": device} if torch.cuda.is_available() else None,
        output_hidden_states=True
    )
    model.eval()

    h17_list = []
    h20_list = []

    with torch.inference_mode():
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx : start_idx + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded, output_hidden_states=True, return_dict=True)
            mask = encoded["attention_mask"].unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)

            # Explicit float32 summation prevents FP16 overflow (>65504)
            h17_raw = outputs.hidden_states[17]
            p17 = (h17_raw * mask).sum(dim=1, dtype=torch.float32) / denom
            h17_list.append(p17.cpu().numpy())

            h20_raw = outputs.hidden_states[20]
            p20 = (h20_raw * mask).sum(dim=1, dtype=torch.float32) / denom
            h20_list.append(p20.cpu().numpy())

            del outputs, encoded, h17_raw, h20_raw

    h17_all = np.nan_to_num(np.vstack(h17_list), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    h20_all = np.nan_to_num(np.vstack(h20_list), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    assert np.isfinite(h17_all).all(), "Layer 17 embeddings contain non-finite values!"
    assert np.isfinite(h20_all).all(), "Layer 20 embeddings contain non-finite values!"
    print(f">> Feature extraction complete: h17 min={h17_all.min():.2f}, max={h17_all.max():.2f}, mean={h17_all.mean():.2f}", flush=True)

    if cache_path:
        np.savez_compressed(cache_path, h17=h17_all, h20=h20_all)
        print(f">> Saved verified cache to {cache_path}", flush=True)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return h17_all, h20_all
