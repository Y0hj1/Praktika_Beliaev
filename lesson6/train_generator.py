
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from generator_transformer import GeneratorTransformer
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.eos_id = tokenizer.token_to_id("<eos>")
        self.tokens = tokenizer.encode(text).ids
        self.samples = [self.tokens[i:i+max_length] for i in range(0, len(self.tokens) - max_length)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        x = x + [self.eos_id]
        x = x[:self.max_length]
        x += [self.pad_id] * (self.max_length - len(x))
        return torch.tensor(x)

def collate_fn(batch):
    return torch.stack(batch)

def train():
    tokenizer = Tokenizer.from_file("mistral_tokenizer.json")
    with open("sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    dataset = TextDataset(text, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = GeneratorTransformer(vocab_size=tokenizer.get_vocab_size()).to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

    model.train()
    for epoch in range(2):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(model.embedding.weight.device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), "generator_model.pt")

if __name__ == "__main__":
    train()
