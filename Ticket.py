import streamlit as st
import pickle

# Load the vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('ticket_priority_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Priority mapping (adjust if needed)
priority_mapping = {0: 'P1', 1: 'P2', 2: 'P3'}

# Optional: Define a clean_text function or just pass raw text
def clean_text(text):
    return text.lower()  # basic lowercase, expand if needed

# Streamlit page config with background image via CSS
st.set_page_config(page_title="🚀 Support Ticket Priority Predictor", page_icon="📩")

# Background image styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1581091215369-9a18f61f832d');
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("🚀 Support Ticket Priority Predictor")

st.write("Paste your ticket description below, and the model will predict if it's P1, P2, or P3!")

# Input box
ticket_text = st.text_area("✉️ Enter Ticket Description:", "")

# Predict button
if st.button("Predict Priority"):
    if ticket_text.strip() == "":
        st.warning("Please enter a ticket description!")
    else:
        cleaned = clean_text(ticket_text)
        vectorized = tfidf.transform([cleaned])
        predicted_class = model.predict(vectorized)[0]
        predicted_priority = priority_mapping[predicted_class]

        st.success(f"✅ **Predicted Priority:** {predicted_priority}")

