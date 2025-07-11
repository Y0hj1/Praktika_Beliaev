
import torch
from tokenizers import Tokenizer
from generator_transformer import GeneratorTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_text(prompt="<bos>", max_len=20):
    tokenizer = Tokenizer.from_file("mistral_tokenizer.json")
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    model = GeneratorTransformer(vocab_size=tokenizer.get_vocab_size()).to(device)
    model.load_state_dict(torch.load("generator_model.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        output_ids = model.generate(input_tensor, max_length=max_len)[0].tolist()

    return tokenizer.decode(output_ids)

if __name__ == "__main__":
    print(generate_text("<bos> Сегодня"))
