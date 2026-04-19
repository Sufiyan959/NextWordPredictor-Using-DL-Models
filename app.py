# import streamlit as st
# import pickle
# import os
# import pandas as pd
# from tensorflow.keras.models import load_model
# from utils.predict import predict_next_word
# from models.transformer_model import TransformerBlock

# # Page config
# st.set_page_config(
#     page_title="Next Word Predictor",
#     page_icon="🤖",
#     layout="centered"
# )

# st.title("Next Word Prediction App")
# st.write("Type a sentence and predict the next word using different models.")

# # ==============================
# # MODEL COMPARISON TABLE
# # ==============================
# st.subheader("Model Comparison")

# data = {
#     "Model": ["RNN", "LSTM", "BiLSTM", "GRU", "Transformer"],
#     "Accuracy": ["Low", "Medium", "High", "High", "Very High"],
#     "Speed": ["Fast", "Medium", "Slow", "Fast", "Medium"],
#     "Memory": ["Low", "Medium", "High", "Medium", "High"],
# }

# df = pd.DataFrame(data)
# st.dataframe(df)

# # ==============================
# # Load tokenizer
# # ==============================
# @st.cache_resource
# def load_tokenizer():
#     with open("saved_models/tokenizer.pkl", "rb") as f:
#         return pickle.load(f)

# # Load config
# @st.cache_resource
# def load_config():
#     with open("saved_models/config.pkl", "rb") as f:
#         return pickle.load(f)

# # Load model dynamically
# @st.cache_resource
# def load_selected_model(model_name):
#     model_path = os.path.join("saved_models", f"{model_name}.h5")

#     if model_name == "transformer":
#         return load_model(
#             model_path,
#             custom_objects={"TransformerBlock": TransformerBlock}
#         )
#     else:
#         return load_model(model_path)

# # Load shared resources
# tokenizer = load_tokenizer()
# config = load_config()
# max_len = config["max_len"]

# # ==============================
# # Model selection
# # ==============================
# model_name = st.selectbox(
#     "Choose Model",
#     ["lstm", "rnn", "bilstm", "gru", "transformer"]
# )

# # Load selected model
# model = load_selected_model(model_name)

# # ==============================
# # User input
# # ==============================
# user_input = st.text_input("Enter your text:")

# # ==============================
# # Prediction
# # ==============================
# if st.button("Predict Next Word"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         try:
#             result = predict_next_word(model, tokenizer, user_input, max_len)
#             st.success(f"Predicted Next Word: **{result}**")
#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# # ==============================
# # SHOW GRAPHS
# # ==============================
# # ==============================
# # SHOW TRAINING GRAPHS
# # ==============================
# st.subheader("📈 Training Graphs")

# model_list = ["rnn", "lstm", "bilstm", "gru", "transformer"]

# for m in model_list:
#     loss_path = f"saved_models/{m}_loss.png"
#     acc_path = f"saved_models/{m}_accuracy.png"

#     if os.path.exists(loss_path) and os.path.exists(acc_path):
#         st.markdown(f"### 🔹 {m.upper()}")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.image(loss_path, caption=f"{m.upper()} Loss")

#         with col2:
#             st.image(acc_path, caption=f"{m.upper()} Accuracy")
#     else:
#         st.warning(f"❌ Graphs not found for {m}")












import streamlit as st
import pickle
import os
from tensorflow.keras.models import load_model
from utils.predict import predict_next_word

# Page config
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Next Word Prediction App")
st.write("Type a sentence and predict the next word using different models.")

# ==============================
# Load tokenizer
# ==============================

@st.cache_resource
def load_tokenizer():
    with open("saved_models/tokenizer.pkl", "rb") as f:
        return pickle.load(f)

# Load config
@st.cache_resource
def load_config():
    with open("saved_models/config.pkl", "rb") as f:
        return pickle.load(f)

# 🔹 Load model
@st.cache_resource
def load_selected_model(model_name):
    model_path = os.path.join("saved_models", f"{model_name}.h5")
    return load_model(model_path)

# 🔹 Load accuracy
@st.cache_resource
def load_metrics(model_name):
    path = f"saved_models/{model_name}_metrics.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ==============================
# 🔹 Load shared resources
# ==============================
tokenizer = load_tokenizer()
config = load_config()
max_len = config["max_len"]

# ==============================
# Model selection
# ==============================

model_name = st.selectbox(
    "Choose Model",
    ["rnn", "lstm", "bilstm", "gru"]
)

# Load model
model = load_selected_model(model_name)

# ==============================
# 📊 Show accuracy
# ==============================
metrics = load_metrics(model_name)

if metrics:
    st.info(f"📊 Accuracy: {metrics['accuracy']:.4f}")
else:
    st.warning("⚠️ Accuracy not available")

# ==============================
# 🔹 Input
# ==============================
user_input = st.text_input("Enter your text:")

# ==============================
# Prediction
# ==============================

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            result = predict_next_word(model, tokenizer, user_input, max_len)
            st.success(f"👉 Predicted Next Word: **{result}**")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ==============================
# 📊 Accuracy comparison table
# ==============================
st.subheader("📊 Model Accuracy Comparison")

model_list = ["rnn", "lstm", "bilstm", "gru"]

acc_data = []

for m in model_list:
    path = f"saved_models/{m}_metrics.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            acc_data.append((m.upper(), data["accuracy"]))

if acc_data:
    import pandas as pd
    df = pd.DataFrame(acc_data, columns=["Model", "Accuracy"])
    st.dataframe(df)

# ==============================
# 📈 SHOW OLD GRAPHS (FROM 40 EPOCHS)
# ==============================
st.subheader("📈 Training Graphs (40 Epochs)")

for m in model_list:
    loss_path = f"saved_models/{m}_loss.png"
    acc_path = f"saved_models/{m}_accuracy.png"

    if os.path.exists(loss_path) and os.path.exists(acc_path):
        st.markdown(f"### 🔹 {m.upper()}")

        col1, col2 = st.columns(2)

        with col1:
            st.image(loss_path, caption=f"{m.upper()} Loss (Old Training)")

        with col2:
            st.image(acc_path, caption=f"{m.upper()} Accuracy (Old Training)")
    else:
        st.warning(f"Graphs not found for {m}")

# ==============================
# 🔹 Footer
# ==============================
st.markdown("---")
st.caption("Built using RNN, LSTM, BiLSTM & GRU ")