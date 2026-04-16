import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# 1. Load dataset
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    return text


# 2. Create tokenizer
def create_tokenizer(text):
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts([text])
    return tokenizer


# 3. Create sequences (same logic as your code)
def create_sequences(text, tokenizer):
    sequences = []

    for sentence in text.split("\n"):
        tokenized = tokenizer.texts_to_sequences([sentence])[0]

        for i in range(1, len(tokenized)):
            n_gram = tokenized[:i+1]
            sequences.append(n_gram)

    return sequences


# 4. Pad sequences
def pad_sequences_data(sequences):
    max_len = max(len(seq) for seq in sequences)
    padded = pad_sequences(sequences, maxlen=max_len, padding='pre')
    return padded, max_len


# 5. Split into X and y
def split_data(padded_sequences):
    X = padded_sequences[:, :-1]
    y = padded_sequences[:, -1]
    return X, y


# 6. One-hot encoding for y
def encode_output(y, vocab_size):
    y = to_categorical(y, num_classes=vocab_size)
    return y


# 7. Complete preprocessing pipeline (MAIN FUNCTION)
def preprocess_pipeline(file_path):
    # Load
    text = load_data(file_path)

    # Tokenizer
    tokenizer = create_tokenizer(text)
    vocab_size = len(tokenizer.word_index) + 1

    # Sequences
    sequences = create_sequences(text, tokenizer)

    # Padding
    padded, max_len = pad_sequences_data(sequences)

    # Split
    X, y = split_data(padded)

    # Encode
    y = encode_output(y, vocab_size)

    return X, y, tokenizer, max_len, vocab_size