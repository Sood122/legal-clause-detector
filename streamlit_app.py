from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import torch

# ✅ Replace restricted model with public one
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Dummy clause labels
id2label = {
    0: "Confidentiality",
    1: "Termination",
    2: "Governing Law",
    3: "Liability",
    4: "Payment Terms"
}

def predict_clause(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=-1)
    return id2label[pred.item()]

# Streamlit App UI
st.set_page_config(page_title="Legal Clause Detector", page_icon="📜")
st.title("📜 Legal Clause Detector")
st.write("Paste a clause below and detect its type.")

clause = st.text_area("Enter legal clause:")

if st.button("Detect"):
    if clause.strip() == "":
        st.warning("Please enter a clause to analyze.")
    else:
        result = predict_clause(clause)
        st.success(f"Detected Clause Type: **{result}**")
