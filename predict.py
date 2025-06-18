#python310 
#predict.py na modeli iz train.py ~ 

import sys, os, time, hashlib, requests, hmac, struct, base58
import torch
import concurrent.futures as futures
from nacl import signing
from datetime import datetime
from train import FullTransformerModel
from constants import IDX2WORD, WORD2IDX, PAD_IDX, BOS_IDX
from bip_utils import (
    Bip39MnemonicValidator, Bip39SeedGenerator,
    Bip44, Bip44Coins, Bip44Changes,
    Bip84, Bip84Coins,
    Bip39Languages, Bip39WordsNum, Bip39MnemonicGenerator,
)
from eth_account import Account

DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH   = "Models/transformer_checkpoint.pt"
ETH_RPC_URL       = os.getenv("ETH_RPC_URL", "https://rpc.ankr.com/eth/84ca8aa47c21b1dbff3d1b5bfd531462a19a11cf9bf675cb084427c54507c625")
SOL_RPC_URL       = os.getenv("SOL_RPC_URL", "https://api.mainnet-beta.solana.com")

def encode_address(addr: str):
    addr_sha = hashlib.sha256(addr.encode("utf-8")).digest()
    addr_tensor = torch.tensor(list(addr_sha), dtype=torch.float32).to(DEVICE)
    return addr_tensor

WORDS_EN = Bip39MnemonicGenerator(lang=Bip39Languages.ENGLISH).FromWordsNumber(Bip39WordsNum.WORDS_NUM_12).ToStr().split()

def fix_checksum(first11: str):
    for w in WORDS_EN:
        full = f"{first11} {w}"
        if Bip39MnemonicValidator().IsValid(full):
            return full
    return None

def generate_with_model(model, addr_tensor, num_variants=5):
    model.eval()
    results = []
    with torch.no_grad():
        for _ in range(num_variants):
            tgt = torch.full((1, 12), PAD_IDX, dtype=torch.long).to(DEVICE)
            tgt[0, 0] = BOS_IDX
            for i in range(1, 12):
                logits, _ = model(addr_tensor, tgt)
                next_token = torch.argmax(logits[0, i - 1])
                tgt[0, i] = next_token
            phrase = " ".join(IDX2WORD[t.item()] for t in tgt[0])
            if Bip39MnemonicValidator().IsValid(phrase):
                results.append(phrase)
            else:
                fixed = fix_checksum(" ".join(phrase.split()[:11]))
                if fixed:
                    results.append(fixed)
    return list(set(results))


def seed_from_mnemonic(m):
    return Bip39SeedGenerator(m).Generate()

def derive_eth(seed):
    w = Bip44.FromSeed(seed, Bip44Coins.ETHEREUM).Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
    key = w.PrivateKey().Raw().ToHex()
    addr = Account.from_key(key).address
    return addr, key

def derive_btc(seed):
    w = Bip84.FromSeed(seed, Bip84Coins.BITCOIN).Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
    key = w.PrivateKey().Raw().ToHex()
    addr = w.PublicKey().ToAddress()
    return addr, key

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
    sk     = signing.SigningKey(priv32)
    pubkey = sk.verify_key.encode()
    secret64 = (sk.encode() + pubkey).hex()
    return base58.b58encode(pubkey).decode(), secret64

def eth_balance(addr):
    try:
        wei = int(
            requests.post(
                ETH_RPC_URL,
                json={"jsonrpc":"2.0","method":"eth_getBalance","params":[addr,"latest"],"id":1},
                timeout=8,
            ).json()["result"],
            16,
        )
        return wei / 1e18
    except:
        return None

def sol_balance(addr):
    try:
        resp = requests.post(
            SOL_RPC_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [addr]},
            timeout=8,
        )
        lamports = resp.json()["result"]["value"]
        return lamports / 1e9
    except:
        return None
    
def main():
    if len(sys.argv) < 2:
        try:
            addr = input("Enter address (or list via space): ").strip()
            if not addr:
                print("No address provided.")
                return
            addresses = addr.split()
        except EOFError:
            print("Input aborted.")
            return
    else:
        addresses = [a for a in sys.argv[1:] if len(a.strip()) >= 5]

    if not addresses:
        print("No valid addresses."); return

    model = FullTransformerModel().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model"])
    model.eval(); print("\u2705 Model loaded.")

    for addr in addresses:
        print(f"\n\ud83d\udd0d {addr}")
        phrases = generate_with_model(model, encode_address(addr))
        if not phrases:
            print("No valid mnemonics"); continue

        for ph in phrases:
            seed = seed_from_mnemonic(ph)
            eth_a, _ = derive_eth(seed)
            btc_a, _ = derive_btc(seed)
            sol_a, _ = derive_sol(seed)
            print("\n\ud83d\udd11", ph)
            print(f"ETH: {eth_a}, Bal: {eth_balance(eth_a)}")
            print(f"BTC: {btc_a}")
            print(f"SOL: {sol_a}, Bal: {sol_balance(sol_a)}")

if __name__ == "__main__":
    main()
