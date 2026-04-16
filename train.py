import os
import pickle

from preprocessing.preprocess import preprocess_pipeline

# CHANGE MODEL HERE
MODEL_NAME = "rnn"   # rnn, lstm, bilstm, gru, transformer


# Import model dynamically
if MODEL_NAME == "rnn":
    from models.rnn_model import build_model
elif MODEL_NAME == "lstm":
    from models.lstm_model import build_model
elif MODEL_NAME == "bilstm":
    from models.bilstm_model import build_model
elif MODEL_NAME == "gru":
    from models.gru_model import build_model
elif MODEL_NAME == "transformer":
    from models.transformer_model import build_model
else:
    raise ValueError("Invalid model name")


def main():

    print("🔹 Loading and preprocessing data...")

    X, y, tokenizer, max_len, vocab_size = preprocess_pipeline("data/dataset.txt")

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Max Sequence Length: {max_len}")

    print(f"🔹 Building {MODEL_NAME.upper()} model...")

    model = build_model(vocab_size, max_len)
    model.summary()

    print("Training started...")

    # Train model
    history = model.fit(
        X,
        y,
        epochs=25,#RNN → 25,LSTM → 45,BiLSTM → 55,GRU → 45,Transformer → 60
        batch_size=64
    )

    # Create folder
    os.makedirs("saved_models", exist_ok=True)

    # =========================
    # SAVE MODEL
    # =========================
    model_path = f"saved_models/{MODEL_NAME}.h5"
    model.save(model_path)
    print(f"Model saved at {model_path}")

    # =========================
    # SAVE TOKENIZER
    # =========================
    with open("saved_models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("Tokenizer saved")

    # =========================
    # SAVE CONFIG
    # =========================
    with open("saved_models/config.pkl", "wb") as f:
        pickle.dump({"max_len": max_len}, f)

    print("Config saved")

    # =========================
    # SAVE ACCURACY
    # =========================
    final_acc = history.history['accuracy'][-1]

    with open(f"saved_models/{MODEL_NAME}_metrics.pkl", "wb") as f:
        pickle.dump({"accuracy": float(final_acc)}, f)

    print(f"Accuracy saved: {final_acc}")


if __name__ == "__main__":
    main()



# import os
# import pickle
# import matplotlib.pyplot as plt

# from preprocessing.preprocess import preprocess_pipeline

# CHANGE MODEL HERE
# MODEL_NAME = "gru"   # rnn, lstm, bilstm, gru, transformer


# # 🔹 Import model dynamically
# if MODEL_NAME == "rnn":
#     from models.rnn_model import build_model
# elif MODEL_NAME == "lstm":
#     from models.lstm_model import build_model
# elif MODEL_NAME == "bilstm":
#     from models.bilstm_model import build_model
# elif MODEL_NAME == "gru":
#     from models.gru_model import build_model
# elif MODEL_NAME == "transformer":
#     from models.transformer_model import build_model
# else:
#     raise ValueError("Invalid model name")


# def main():

#     print("🔹 Loading and preprocessing data...")

#     # Load + preprocess
#     X, y, tokenizer, max_len, vocab_size = preprocess_pipeline("data/dataset.txt")

#     print(f"Vocabulary Size: {vocab_size}")
#     print(f"Max Sequence Length: {max_len}")

#     print(f"🔹 Building {MODEL_NAME.upper()} model...")

#     # Build model
#     model = build_model(vocab_size, max_len)
#     model.summary()

#     print("🔹 Training started...")

#     # 🔥 Train model and store history
#     history = model.fit(
#         X,
#         y,
#         epochs=40,# transformer==40,rnn==40,lstm==40
#         batch_size=64
#     )

#     # 🔹 Create folder if not exists
#     os.makedirs("saved_models", exist_ok=True)

#     # =========================
#     # 📉 LOSS GRAPH
#     # =========================
#     loss_path = f"saved_models/{MODEL_NAME}_loss.png"

#     plt.figure()
#     plt.plot(history.history['loss'])
#     plt.title(f'{MODEL_NAME.upper()} Loss vs Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.savefig(loss_path)
#     plt.close()

#     # =========================
#     # 📈 ACCURACY GRAPH
#     # =========================
#     acc_path = f"saved_models/{MODEL_NAME}_accuracy.png"

#     plt.figure()
#     plt.plot(history.history['accuracy'])
#     plt.title(f'{MODEL_NAME.upper()} Accuracy vs Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.savefig(acc_path)
#     plt.close()

#     print(f"📊 Graphs saved: {loss_path}, {acc_path}")

#     # =========================
#     # 💾 SAVE MODEL
#     # =========================
#     model_path = f"saved_models/{MODEL_NAME}.h5"
#     model.save(model_path)
#     print(f"✅ Model saved at {model_path}")

#     # =========================
#     # 💾 SAVE TOKENIZER
#     # =========================
#     with open("saved_models/tokenizer.pkl", "wb") as f:
#         pickle.dump(tokenizer, f)

#     print("✅ Tokenizer saved")

#     # =========================
#     # 💾 SAVE CONFIG
#     # =========================
#     with open("saved_models/config.pkl", "wb") as f:
#         pickle.dump({"max_len": max_len}, f)

#     print("✅ Config saved (max_len)")


# if __name__ == "__main__":
#     main()