import json
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cosine_similarity
from constants import WORD2IDX, IDX2WORD, VOCAB_SIZE, PAD_IDX, BOS_IDX
from bip_utils import Bip39MnemonicGenerator, Bip39MnemonicValidator, Bip39SeedGenerator, Bip39WordsNum, Bip44, Bip44Coins, Bip44Changes, Bip84, Bip84Coins
from eth_account import Account
from nacl import signing
import hmac, struct, base58, os, random
from datetime import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHRASE_LEN = 12
ADDR_LEN = 32

TARGET_PHRASE = "***"
TARGET_ETH = "***"
TARGET_SOL = "***"
TARGET_BTC = "***"

REFERENCE_PHRASES = {
    tuple(TARGET_PHRASE.split()): 5.0,
    tuple("***".split()): 5.0,
}

RECALL_SCORE_MAP = {
    1: 0.001, 2: 0.002, 3: 0.003, 4: 0.005, 5: 0.008, 6: 0.015,
    7: 0.025, 8: 0.030, 9: 0.035, 10: 0.040, 11: 0.045, 12: 0.05
}

def derive_path_ed25519(path: str, seed: bytes) -> bytes:
    digest = hmac.new(b"ed25519 seed", seed, hashlib.sha512).digest()
    k, c = digest[:32], digest[32:]
    for level in path.strip().split("/")[1:]:
        hardened = level.endswith("'")
        idx = int(level.rstrip("'")) + (0x80000000 if hardened else 0)
        idx_bytes = struct.pack(">I", idx)
        data = (b"\x00" + k if hardened else signing.SigningKey(k).verify_key.encode()) + idx_bytes
        digest = hmac.new(c, data, hashlib.sha512).digest()
        k, c = digest[:32], digest[32:]
    return k

def derive_sol(seed, path="m/44'/501'/0'/0'"):
    priv32 = derive_path_ed25519(path, seed)
    sk = signing.SigningKey(priv32)
    pubkey = sk.verify_key.encode()
    secret64 = (sk.encode() + pubkey).hex()
    return base58.b58encode(pubkey).decode(), secret64

def derive_eth(seed):
    w = Bip44.FromSeed(seed, Bip44Coins.ETHEREUM).Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
    key = w.PrivateKey().Raw().ToHex()
    return Account.from_key(key).address

def derive_btc(seed):
    w = Bip84.FromSeed(seed, Bip84Coins.BITCOIN).Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
    return w.PublicKey().ToAddress()

def is_valid_unique_phrase(phrase, validator):
    return validator.IsValid(" ".join(phrase)) and len(set(phrase)) == len(phrase)

def generate_valid_phrase(generator, validator):
    while True:
        mnemonic = generator.FromWordsNumber(Bip39WordsNum.WORDS_NUM_12)
        phrase = str(mnemonic).split()
        if is_valid_unique_phrase(phrase, validator):
            return phrase

class MnemonicDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    phrase = item["mnemonic"].split()
                    if all(word in WORD2IDX for word in phrase):
                        for coin in ["eth", "btc", "sol"]:
                            address = item.get(coin)
                            if address:
                                self.data.append((address, phrase))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(self.data):,} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        address, phrase = self.data[idx]
        sha = hashlib.sha256(address.encode("utf-8")).digest()
        addr_bytes = torch.frombuffer(bytearray(sha), dtype=torch.uint8).to(torch.long)
        phrase_tensor = torch.tensor([WORD2IDX[word] for word in phrase], dtype=torch.long)
        weight = REFERENCE_PHRASES.get(tuple(w.strip().lower() for w in phrase), 1.0)
        return addr_bytes, phrase_tensor, torch.tensor(weight, dtype=torch.float)

class FullTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_phrase = nn.Embedding(VOCAB_SIZE, 128, padding_idx=PAD_IDX)
        self.pos_enc_phrase = nn.Parameter(torch.randn(1, PHRASE_LEN, 128))
        self.byte_embed = nn.Embedding(256, 64)
        self.pos_enc_addr = nn.Parameter(torch.randn(1, ADDR_LEN, 64))
        self.addr_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256, batch_first=True),
            num_layers=4
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True),
            num_layers=8
        )
        self.proj_addr = nn.Linear(64, 128)
        self.fc = nn.Linear(128, VOCAB_SIZE)
        self.dropout = nn.Dropout(0.2)

    def forward(self, addr_bytes, phrase_input):
        phrase_embed = self.embed_phrase(phrase_input) + self.pos_enc_phrase[:, :phrase_input.size(1), :]
        phrase_embed = self.dropout(phrase_embed)
        addr_embed = self.byte_embed(addr_bytes) + self.pos_enc_addr[:, :addr_bytes.size(1), :]
        addr_encoded = self.addr_encoder(addr_embed)
        addr_memory = self.proj_addr(addr_encoded)
        output = self.decoder(tgt=phrase_embed, memory=addr_memory)
        logits = self.fc(output)
        return logits, output.mean(dim=1)

def inject_synthetic_batch(model, optimizer, criterion, num_samples=64):
    model.train()
    generator = Bip39MnemonicGenerator()
    validator = Bip39MnemonicValidator()

    for _ in range(num_samples):
        phrase = generate_valid_phrase(generator, validator)

        try:
            mnemonic = " ".join(phrase)
            seed = Bip39SeedGenerator(mnemonic).Generate()
            eth = derive_eth(seed)
            btc = derive_btc(seed)
            sol, _ = derive_sol(seed)

            for addr in [eth, btc, sol]:
                sha = hashlib.sha256(addr.encode("utf-8")).digest()
                addr_bytes = torch.tensor([list(sha)], dtype=torch.long).to(DEVICE)
                phrase_tensor = torch.tensor([[WORD2IDX[w] for w in phrase]], dtype=torch.long).to(DEVICE)

                recall_n = random.randint(1, PHRASE_LEN)
                decoder_input = torch.full((1, recall_n), PAD_IDX, dtype=torch.long).to(DEVICE)
                decoder_input[0][0] = BOS_IDX
                decoder_input[0, 1:recall_n] = phrase_tensor[0, :recall_n - 1]
                target_output = phrase_tensor[:, :recall_n]

                logits, _ = model(addr_bytes, decoder_input)
                loss_all = criterion(logits.view(-1, VOCAB_SIZE), target_output.view(-1)).view(target_output.shape)
                loss_ce = loss_all.sum()
                loss_ce.backward()
                optimizer.step()
                optimizer.zero_grad()
        except Exception:
            continue

def check_target_recovery(model):
    model.eval()
    validator = Bip39MnemonicValidator()
    with torch.no_grad():
        for target_addr in [TARGET_ETH, TARGET_BTC, TARGET_SOL]:
            sha = hashlib.sha256(target_addr.encode("utf-8")).digest()
            addr_bytes = torch.tensor([list(sha)], dtype=torch.long).to(DEVICE)
            decoder_input = torch.full((1, PHRASE_LEN), PAD_IDX, dtype=torch.long).to(DEVICE)
            decoder_input[0][0] = BOS_IDX

            used_tokens = set()
            for i in range(1, PHRASE_LEN):
                logits, _ = model(addr_bytes, decoder_input[:, :i])
                mask = torch.full((VOCAB_SIZE,), 0.0, device=DEVICE)
                for tok in used_tokens:
                    mask[tok] = -float('inf')
                next_token = (logits[0, i - 1] + mask).argmax().item()
                used_tokens.add(next_token)
                decoder_input[0][i] = next_token

            pred_phrase = [IDX2WORD[idx] for idx in decoder_input[0].tolist()]
            print(f"🔎 Trying to recover phrase for: {target_addr}")
            print("Predicted:", " ".join(pred_phrase))

            if not is_valid_unique_phrase(pred_phrase, validator):
                print("Phrase is invalid or has duplicates.")
                continue

            print("Target:   ", TARGET_PHRASE)
            if " ".join(pred_phrase).strip().lower() == TARGET_PHRASE.strip().lower():
                return True
    return False

def train():
    dataset = MnemonicDataset("Data/dataset.jsonl")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    model = FullTransformerModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1, reduction="none")

    best_acc = 0
    checkpoint_path = "Models/transformer_checkpoint.pt"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0)
        print(f"🔁 Resuming from epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, 10000):
            model.train()
            total_loss, total_acc, count = 0, 0, 0

            for addr_bytes, phrase_tensor, weights in tqdm(dataloader, desc=f"Epoch {epoch}"):
                addr_bytes = addr_bytes.to(DEVICE, non_blocking=True)
                phrase_tensor = phrase_tensor.to(DEVICE, non_blocking=True)
                weights = weights.to(DEVICE)

                optimizer.zero_grad()
                recall_n = random.choice(list(RECALL_SCORE_MAP.keys()))
                decoder_input = torch.full_like(phrase_tensor[:, :recall_n], PAD_IDX)
                decoder_input[:, 0] = BOS_IDX
                decoder_input[:, 1:] = phrase_tensor[:, :recall_n - 1]
                target_output = phrase_tensor[:, :recall_n]

                logits, features = model(addr_bytes, decoder_input)
                loss_all = criterion(logits.reshape(-1, VOCAB_SIZE), target_output.reshape(-1)).reshape(target_output.shape)

                pred = logits.argmax(dim=-1)
                match = (pred == target_output).all(dim=1).float()
                recall_score = RECALL_SCORE_MAP[recall_n]
                total_acc += (match.mean().item() * recall_score)

                loss_ce = (loss_all.sum(dim=1) * weights).mean()

                shuffled = features[torch.randperm(features.size(0))]
                cosine = cosine_similarity(features, shuffled, dim=-1)
                contrastive = (1 - cosine).mean()

                loss = loss_ce + 0.2 * contrastive
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / count
            avg_acc = total_acc / count

            if epoch % 10 == 0:
                print("Attempting to recover TARGET_PHRASE...")
                success = check_target_recovery(model)
                if success:
                    avg_acc += 0.01
                    print("Target phrase recovered. Boosting accuracy.")
                else:
                    print("Target phrase NOT recovered.")

            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")
            scheduler.step()

            if epoch % 5 == 0:
                print("Injecting synthetic mnemonics...")
                inject_synthetic_batch(model, optimizer, criterion)

            if avg_acc > best_acc:
                best_acc = avg_acc
                print("✅ Saving best model...")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc
                }, checkpoint_path)

    except KeyboardInterrupt:
        print("Interrupted. Saving checkpoint...")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc
        }, checkpoint_path)
        print("Checkpoint saved.")

if __name__ == "__main__":
    print("Starting training...")
    train()
