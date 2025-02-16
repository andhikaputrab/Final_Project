import streamlit as st
from src.utils.styling import load_css

load_css()

st.title("🌱 Welcome to the Biology Question Answering App")
st.markdown("""
🔍 Find Quick & Accurate Answers to Your Biology QuestionsWhether you're a student studying biology or simply curious about the natural world, this app is here to help you discover clear and insightful answers to your biology-related questions. Just type in your question, and let the app provide easy-to-understand explanations backed by reliable sources.

📚 Explore Biology Topics:
- 🌿 Ecology and Environment
- 🧬 Genetics and Evolution
- 🧠 Anatomy and Physiology
- 🔬 Microbiology and Biotechnology
- 🌍 Life on Earth and Beyond

This app is designed for all users, from students to professionals, looking to dive deeper into various biological concepts. We’re committed to making biology learning more fun and interactive.
""")