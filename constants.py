# constants.py

SPECIAL_TOKENS = ["<PAD>", "<BOS>"]

with open("Data/bip39.txt", "r", encoding="utf-8") as f:
    BIP39 = [line.strip() for line in f if line.strip()]

BIP39 = SPECIAL_TOKENS + BIP39

WORD2IDX = {word: idx for idx, word in enumerate(BIP39)}
IDX2WORD = {idx: word for word, idx in WORD2IDX.items()}
VOCAB_SIZE = len(BIP39)
PAD_IDX = WORD2IDX["<PAD>"]
BOS_IDX = WORD2IDX["<BOS>"]
