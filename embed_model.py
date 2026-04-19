from transformers import AutoTokenizer, AutoModel
import torch

model_name = "Qwen/Qwen3-Embedding-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()

def embed_text(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state

    mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()

    embeddings = torch.sum(last_hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    return embeddings