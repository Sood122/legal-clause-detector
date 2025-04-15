import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained model and tokenizer from Hugging Face
model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=41)
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Define class-to-label mapping (assuming 41 classes for legal clauses)
id2label = {
    0: 'Clause 1', 1: 'Clause 2', 2: 'Clause 3',  # Add all class names here
    40: 'Clause 41'  # Example, fill in actual classes from the dataset
}

# Function to predict clause type
def predict_clause(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# Streamlit UI
st.title("Legal Clause Detector")
st.markdown("Enter a contract clause below to detect its type:")

user_input = st.text_area("Enter Contract Clause:", "")
if st.button("Detect Clause"):
    if user_input:
        prediction = predict_clause(user_input)
        clause_type = id2label.get(prediction, "Unknown")
        st.success(f"Predicted Clause Type: {clause_type}")
    else:
        st.error("Please enter a contract clause.")
