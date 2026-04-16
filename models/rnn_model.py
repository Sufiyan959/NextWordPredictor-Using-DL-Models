from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_model(vocab_size, max_len):
    model = Sequential()

    # Embedding Layer
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=128,
        input_length=max_len - 1
    ))

    # RNN Layer
    model.add(SimpleRNN(
        128,
        return_sequences=False
    ))

    # Dropout (prevent overfitting)
    model.add(Dropout(0.3))

    # Output Layer
    model.add(Dense(
        vocab_size,
        activation='softmax'
    ))

    # Compile
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model