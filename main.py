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
        return [stoi[c] for c in s if c in stoi]

    def decode(tokens):
        return "".join([itos[i] for i in tokens])

    return encode, decode, vocab_size


def load_model(vocab_size):
    model = GPTLanguageModel(vocab_size)

    model.load_state_dict(
        torch.load(
            "model.pt",
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


def main():

    encode, decode, vocab_size = load_encoder_decoder()

    print("Loading model...")
    model = load_model(vocab_size)
    print("Done!")

    print("=" * 60)
    print("NanoGPT Interactive")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:

        prompt = input("\nYou: ").strip()

        if prompt.lower() == "exit":
            break

        if not prompt:
            continue

        encoded = encode(prompt)

        if len(encoded) == 0:
            print("Prompt contains no valid characters.")
            continue

        context = torch.tensor(
            encoded,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        print("\nModel: ", end="", flush=True)

        for token in model.stream_generate(
            context,
            max_new_tokens=300,
            temperature=0.8,
            top_k=40,
        ):
            print(decode([token]), end="", flush=True)

        print("\n")


if __name__ == "__main__":
    main()