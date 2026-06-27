import torch
from model import GPTLanguageModel

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("dataset/input.txt", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}

encode = lambda s:[stoi[c] for c in s]

data = torch.tensor(encode(text))

# your batching code here...

model = GPTLanguageModel(vocab_size).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

# -------------------------
# Your entire training loop
# -------------------------

torch.save(model.state_dict(), "model.pt")
print("Saved.")