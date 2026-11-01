
import os
import re
import json
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
import gradio as gr

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer, util


# ------------------------------------------
# 1. Project file paths
# ------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "SRKR_detailed_chatbot_dataset_v2.json"
MODEL_PATH = BASE_DIR / "srkr_lstm_model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pickle"
ENCODER_PATH = BASE_DIR / "label_encoder.pickle"


# ------------------------------------------
# 2. Load chatbot files
# ------------------------------------------

print("Loading SRKR chatbot files...")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset_json = json.load(f)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

model = load_model(MODEL_PATH, compile=False)

# Get sequence length directly from trained model
max_len = int(model.input_shape[-1])

# Load knowledge base
kb_records = dataset_json.get("knowledge_base", [])

if not kb_records:
    raise ValueError("Knowledge base is empty.")

# ------------------------------------------
# 3. Text preprocessing
# ------------------------------------------

def preprocess_text(text):
    if not text:
        return ""

    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------------------
# 4. Initialize semantic retrieval
# ------------------------------------------

print("Loading Sentence Transformer...")

retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")

kb_passages = []

for item in kb_records:
    passage = (
        f"{item.get('title', '')} "
        f"({item.get('category', '')}): "
        f"{item.get('answer', '')}"
    )
    kb_passages.append(passage)

kb_embeddings = retrieval_model.encode(
    kb_passages,
    convert_to_tensor=True
)

print("SRKR chatbot loaded successfully!")


# ------------------------------------------
# 5. Chatbot response function
# ------------------------------------------

def get_bot_response(user_query):

    if not user_query or not user_query.strip():
        return "Please enter a valid question.", "Empty Query", 0.0, None

    cleaned = preprocess_text(user_query)

    seq = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        seq,
        maxlen=max_len,
        padding="post"
    )

    # LSTM intent classification
    pred = model.predict(padded, verbose=0)[0]

    max_idx = int(np.argmax(pred))
    confidence = float(pred[max_idx])

    predicted_tag = label_encoder.classes_[max_idx]

    # Semantic retrieval
    query_embedding = retrieval_model.encode(
        user_query,
        convert_to_tensor=True
    )

    cos_scores = util.cos_sim(
        query_embedding,
        kb_embeddings
    )[0]

    best_match_idx = int(tf.argmax(cos_scores).numpy())
    best_score = float(cos_scores[best_match_idx].numpy())

    # Fallback response
    if confidence < 0.25 and best_score < 0.30:

        fallback_answer = (
            "I'm sorry, I cannot find an exact answer to your "
            "query in our SRKR records. Please visit the official "
            "college website: https://www.srkrec.ac.in"
        )

        return fallback_answer, "Fallback / Unknown", confidence, None

    # Retrieve matching answer
    matched_kb = kb_records[best_match_idx]

    answer = matched_kb.get("answer", "")
    sources = matched_kb.get("source_urls", [])

    return answer, predicted_tag, confidence, sources


# ------------------------------------------
# 6. Gradio chatbot interface
# ------------------------------------------

def chat_wrapper(message, history):

    ans, tag, conf, sources = get_bot_response(message)

    formatted_response = str(ans)

    if sources:
        formatted_response += (
            "\n\n**Official Sources:**\n"
            + "\n".join([f"- [{s}]({s})" for s in sources])
        )

    return formatted_response


custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate"
)

custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto;
}

.header-box {
    text-align: center;
    background-color: #0d47a1;
    color: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.header-box h1 {
    margin: 0;
    color: white !important;
    font-size: 26px;
}

.header-box p {
    margin: 5px 0 0 0;
    color: #e3f2fd;
    font-size: 14px;
}
"""


with gr.Blocks(
    theme=custom_theme,
    css=custom_css
) as demo:

    gr.HTML("""
    <div class='header-box'>
        <h1>SRKR Engineering College Chatbot</h1>
        <p>
        AI-Powered College Information Assistant
        (LSTM & Semantic Retrieval)
        </p>
    </div>
    """)

    gr.ChatInterface(
        fn=chat_wrapper,
        examples=[
            "What courses are offered at SRKR?",
            "What are the admission procedures?",
            "Tell me about the college hostel.",
            "What library facilities are available?",
            "How can I get placement information?",
            "Where can I find the R23 syllabus?",
            "Does the college provide transportation?"
        ]
    )


# ------------------------------------------
# 7. Launch web application
# ------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
