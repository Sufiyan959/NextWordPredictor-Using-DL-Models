import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Reverse index
def build_reverse_index(tokenizer):
    return {index: word for word, index in tokenizer.word_index.items()}


# Predict next word (ONLY FUNCTION YOU NEED)
def predict_next_word(model, tokenizer, text, max_len):
    
    # Convert text → sequence
    tokenized = tokenizer.texts_to_sequences([text])[0]

    if len(tokenized) == 0:
        return "Unknown input"

    # Pad input
    padded = pad_sequences([tokenized], maxlen=max_len - 1, padding='pre')

    # Predict
    preds = model.predict(padded, verbose=0)
    predicted_index = np.argmax(preds)

    # Convert index → word
    reverse_index = build_reverse_index(tokenizer)

    return reverse_index.get(predicted_index, "Not found")