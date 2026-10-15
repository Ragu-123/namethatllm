import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def extract_qwen_features(texts, model_dir, cache_path=None, batch_size=16, max_length=512):
    """Extract mean-pooled Layer 17 & Layer 20 hidden states from Qwen 3 (1.7B-Base)."""
    if cache_path and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            if np.isfinite(data["h17"]).all() and np.isfinite(data["h20"]).all():
                print(f"Loading verified cached Qwen representations from {cache_path}...", flush=True)
                return data["h17"], data["h20"]
            else:
                print(f"Cache {cache_path} contains non-finite values! Re-extracting in float32...", flush=True)
        except Exception as e:
            print(f"Error reading cache {cache_path}: {e}. Re-extracting...", flush=True)

    print(f"Extracting Qwen 3 representations ({len(texts)} passages, batch_size={batch_size})...", flush=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(
        model_dir,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": device} if torch.cuda.is_available() else None,
        output_hidden_states=True
    )
    model.eval()

    h17_list = []
    h20_list = []

    with torch.no_grad():
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx : start_idx + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded, output_hidden_states=True)
            # Cast hidden states and mask to float32 before summation to prevent float16 overflow (>65504)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            denom = torch.clamp(mask.sum(dim=1), min=1.0)

            # Layer 17 mean-pooling in float32
            layer17 = outputs.hidden_states[17].float()
            p17 = (layer17 * mask).sum(dim=1) / denom
            h17_list.append(p17.cpu().numpy())

            # Layer 20 mean-pooling in float32
            layer20 = outputs.hidden_states[20].float()
            p20 = (layer20 * mask).sum(dim=1) / denom
            h20_list.append(p20.cpu().numpy())

    h17_all = np.nan_to_num(np.vstack(h17_list), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    h20_all = np.nan_to_num(np.vstack(h20_list), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if cache_path:
        np.savez_compressed(cache_path, h17=h17_all, h20=h20_all)
        print(f"Verified Qwen representations saved to {cache_path}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return h17_all, h20_all
