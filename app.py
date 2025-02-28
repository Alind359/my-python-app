import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.title("💬 AI Chatbot (Powered by Phi-2)")
st.write("Ask me anything!")

# Load AI model
@st.cache_resource
def load_model():
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer

model, tokenizer = load_model()

# User input
user_input = st.text_input("You:", "")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    st.write("🤖 AI:", response)
