import requests

TARGET_URL = "http://127.0.0.1:30030/generate"


def main():
    prompts = [
        "Explain quantum entanglement in one sentence.",
        "Explain the Transformer architecture in one sentence.",
        "Explain vanishing gradients in one sentence.",
        "Explain overfitting in one sentence.",
        "Explain regularization in one sentence.",
        "Explain the attention mechanism in one sentence.",
        "Explain autoregressive modeling in one sentence.",
        "Explain perplexity (PPL) in one sentence.",
    ]

    payload = {
        "text": prompts,
        "sampling_params": [{"max_new_tokens": 16} for _ in prompts],
        "group_all": [8] * 8,
        "group_id": list(range(8)),
        "group_topk": [4] * 8,
        "group_key": ["demo-group-1"] * 8,
        "stream": False,
    }

    resp = requests.post(TARGET_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    for i, item in enumerate(data):
        text = item.get("text", "")
        meta = item.get("meta_info", {})
        print(f"[{i}] finish_reason={meta.get('finish_reason')}")
        print(text)
        print("-" * 60)


if __name__ == "__main__":
    main()
