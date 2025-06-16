#!/usr/bin/env python3
# predictor.py — Mnemonic phrase predictor with ETH, BTC, SOL derivation and balance checking

import sys, os, time, hashlib, requests, hmac, struct, base58
import torch
import concurrent.futures as futures
from nacl import signing
from datetime import datetime
from train import FullTransformerModel, IDX2WORD, WORD2IDX, PAD_IDX
from bip_utils import (
    Bip39MnemonicValidator, Bip39SeedGenerator,
    Bip44, Bip44Coins, Bip44Changes,
    Bip84, Bip84Coins,
    Bip39Languages, Bip39WordsNum, Bip39MnemonicGenerator,
)
from eth_account import Account

# ───────────────────────── CONFIG ─────────────────────────
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH   = "Models/transformer_checkpoint.pt"
NUM_VARIANTS      = 8
BATCH_PHRASES     = 7
BTC_MAX_WORKERS   = 3
BTC_RETRIES       = 3

ETH_RPC_URL = os.getenv("ETH_RPC_URL", "https://rpc.ankr.com/eth")
SOL_RPC_URL = os.getenv("SOL_RPC_URL", "https://api.mainnet-beta.solana.com")

WORDS_EN = Bip39MnemonicGenerator(lang=Bip39Languages.ENGLISH).FromWordsNumber(Bip39WordsNum.WORDS_NUM_12).ToStr().split()
os.makedirs("Data", exist_ok=True)

# ───────────────────────── UTILS ─────────────────────────

def encode_addresses_batch(addresses):
    hashes = [list(hashlib.sha256(a.encode()).digest()) for a in addresses]
    return torch.tensor(hashes, dtype=torch.long, device=DEVICE)

def encode_address(addr: str):
    return encode_addresses_batch([addr])

def fix_checksum(first11: str):
    for w in WORDS_EN:
        full = f"{first11} {w}"
        if Bip39MnemonicValidator().IsValid(full):
            return full
    return None

# ───────────────────────── GENERATION ─────────────────────────

def generate_phrases(model, addr_tensor, max_phrases=BATCH_PHRASES):
    model.eval(); tried=set(); out=[]
    with torch.no_grad():
        logits0,_ = model(addr_tensor, torch.full((1,12), PAD_IDX, dtype=torch.long, device=DEVICE))
        p0 = torch.softmax(logits0[0,0], dim=0)
        for idx in torch.argsort(p0, descending=True):
            if len(out) >= max_phrases: break
            gen = torch.full((1,12), PAD_IDX, dtype=torch.long, device=DEVICE)
            gen[0,0] = idx
            words=[IDX2WORD[idx.item()]]
            for pos in range(1,11):
                logits,_ = model(addr_tensor, gen.clone())
                nxt=torch.argmax(torch.softmax(logits[0,pos], dim=0))
                gen[0,pos]=nxt; words.append(IDX2WORD[nxt.item()])
            raw=" ".join(words)
            fixed=fix_checksum(raw)
            if fixed and fixed not in tried:
                tried.add(fixed); out.append(fixed)
    return out

# ───────────────────────── DERIVATIONS ─────────────────────────

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

# ───────────────────────── BALANCE CHECKS ─────────────────────────

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

def btc_balance_single(addr: str, retries: int = BTC_RETRIES):
    url = f"https://blockstream.info/api/address/{addr}"
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                funded = data["chain_stats"]["funded_txo_sum"]
                spent  = data["chain_stats"]["spent_txo_sum"]
                return (funded - spent) / 1e8
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
    return None

# ───────────────────────── MAIN ─────────────────────────

def main():
    if len(sys.argv) < 2:
        addr = input("Enter address (or list via space): ").strip()
        if not addr: return
        addresses = addr.split()
    else:
        addresses = [a for a in sys.argv[1:] if len(a.strip()) >= 5]

    if not addresses:
        print("No valid addresses."); return

    model = FullTransformerModel().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model"])
    model.eval(); print("✅ Model loaded.")

    for addr in addresses:
        print(f"\n🔍 {addr}")
        phrases = generate_phrases(model, encode_address(addr), BATCH_PHRASES)
        if not phrases:
            print("No valid mnemonics"); continue

        btc_jobs = []
        for ph in phrases:
            seed = seed_from_mnemonic(ph)
            eth_a, eth_k = derive_eth(seed)
            btc_a, btc_k = derive_btc(seed)
            sol_a, sol_k = derive_sol(seed)

            eb = eth_balance(eth_a)
            sb = sol_balance(sol_a)

            btc_jobs.append((btc_a, ph, eth_a, eb, sol_a, sb))

        with futures.ThreadPoolExecutor(max_workers=BTC_MAX_WORKERS) as ex:
            fut_to_data = {ex.submit(btc_balance_single, j[0]): j for j in btc_jobs}
            for fut in futures.as_completed(fut_to_data):
                bb = fut.result()
                btc_a, ph, eth_a, eb, sol_a, sb = fut_to_data[fut]

                print("\n🔑", ph)
                print(f"ETH: {eth_a}, Bal: {eb}")
                print(f"BTC: {btc_a}, Bal: {bb}")
                print(f"SOL: {sol_a}, Bal: {sb}")

                with open("attempts.txt", "a", encoding="utf-8") as logf:
                    logf.write(f"[{datetime.now().isoformat()}] TRY | Phrase: {ph}\n")
                    logf.write(f"    ETH: {eth_a}, Bal: {eb}\n")
                    logf.write(f"    BTC: {btc_a}, Bal: {bb}\n")
                    logf.write(f"    SOL: {sol_a}, Bal: {sb}\n\n")

if __name__ == "__main__":
    main()