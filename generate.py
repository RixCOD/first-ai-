import torch
from model import MiniGPT
from dataset import CharDataset

# Load dataset & model
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Initialize dataset and model
dataset = CharDataset(text)
model = MiniGPT(dataset.vocab_size)

# Set block_size used during training (must match your model config)
model.block_size = 128  # <-- adjust if your model used a different context length

# Load trained weights
model.load_state_dict(torch.load("rix.pt", map_location=torch.device("cpu")))
model.eval()

def generate(prompt, max_new_tokens=100):
    # Convert prompt to tensor
    idx = torch.tensor([[dataset.char2idx.get(c, 0) for c in prompt]], dtype=torch.long)

    block_size = model.block_size

    # Truncate prompt if it exceeds block size
    if idx.size(1) > block_size:
        idx = idx[:, -block_size:]

    for _ in range(max_new_tokens):
        # Condition on only the last block_size tokens
        idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx

        with torch.no_grad():
            logits = model(idx_cond)

        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

    # Decode generated tokens
    out = ''.join([dataset.idx2char[i.item()] for i in idx[0]])
    return out
