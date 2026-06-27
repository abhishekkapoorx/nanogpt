import torch
from model import GPTLanguageModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder_decoder():
    with open("dataset/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        # Ignore characters that weren't seen during training
        return [stoi[c] for c in s if c in stoi]

    def decode(tokens):
        return "".join([itos[i] for i in tokens])

    return encode, decode, vocab_size


def load_model(vocab_size):
    model = GPTLanguageModel(vocab_size)

    state_dict = torch.load(
        "model.pt",
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def generate_text(model, prompt, encode, decode, max_new_tokens=300):
    encoded = encode(prompt)

    if len(encoded) == 0:
        return "Prompt contains no valid characters from the training vocabulary."

    context = torch.tensor(
        encoded,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_new_tokens
        )

    new_tokens = generated[0][len(encoded):]
    return decode(new_tokens.tolist())


def main():

    encode, decode, vocab_size = load_encoder_decoder()

    print("Loading model...")
    model = load_model(vocab_size)
    print("Model loaded successfully!\n")

    print("=" * 60)
    print("NanoGPT Interactive Chat")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:

        prompt = input("\nYou: ").strip()

        if prompt.lower() == "exit":
            break

        if prompt == "":
            continue

        response = generate_text(
            model,
            prompt,
            encode,
            decode,
            max_new_tokens=300
        )

        print("\nModel:")
        print(response)


if __name__ == "__main__":
    main()