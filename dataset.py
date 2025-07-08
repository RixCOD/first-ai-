# dataset.py
import torch

class CharDataset:
    def __init__(self, text, block_size=128):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.block_size = block_size

        data = torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)
        self.data = data

    def get_batch(self, batch_size=32):
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x, y
