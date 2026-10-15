import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def extract_qwen_features(texts, model_dir, cache_path=None, batch_size=16, max_length=512):
    """Extract mean-pooled Layer 17 & Layer 20 hidden states from Qwen 3 (1.7B-Base)."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached Qwen representations from {cache_path}...", flush=True)
        data = np.load(cache_path)
        return data["h17"], data["h20"]

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
            mask = encoded["attention_mask"].unsqueeze(-1)

            # Layer 17 mean-pooling
            layer17 = outputs.hidden_states[17]
            p17 = (layer17 * mask).sum(dim=1) / mask.sum(dim=1)
            h17_list.append(p17.cpu().to(torch.float32).numpy())

            # Layer 20 mean-pooling
            layer20 = outputs.hidden_states[20]
            p20 = (layer20 * mask).sum(dim=1) / mask.sum(dim=1)
            h20_list.append(p20.cpu().to(torch.float32).numpy())

    h17_all = np.vstack(h17_list)
    h20_all = np.vstack(h20_list)

    if cache_path:
        np.savez_compressed(cache_path, h17=h17_all, h20=h20_all)
        print(f"Cached Qwen representations saved to {cache_path}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return h17_all, h20_all
