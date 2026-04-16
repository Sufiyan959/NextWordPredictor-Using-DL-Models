import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, Dense, Dropout,
    LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Simple Transformer Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])

        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=False):
        # Attention
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)

        # Skip connection + normalization
        x = self.norm1(inputs + attn_output)

        # Feed forward
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)

        # Final output
        return self.norm2(x + ffn_output)


# Build Model
def build_model(vocab_size, max_len):

    embed_dim = 128      # word vector size
    num_heads = 4        # attention heads
    ff_dim = 256         # feedforward size

    inputs = tf.keras.Input(shape=(max_len - 1,))

    # Embedding
    x = Embedding(vocab_size, embed_dim)(inputs)

    # Transformer Block
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    # Pool sequence -> vector
    x = GlobalAveragePooling1D()(x)

    # Dropout
    x = Dropout(0.3)(x)

    # Output layer
    outputs = Dense(vocab_size, activation="softmax")(x)

    model = Model(inputs, outputs)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model