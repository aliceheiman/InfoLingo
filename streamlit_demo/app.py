import streamlit as st

from infolingo import Infolingo, tokenize_text


@st.cache_data
def get_infolingo(language: str = "english"):
    return Infolingo(language)


st.image("assets/banner.png")
st.title("👋🌎 Welcome to InfoLingo!")
st.info(
    "Infolingo uses information theory to pick best words to learn given a piece of foreign language text you want to understand. "
)

language_choice = st.radio("Pick a Language", ["English", "Spanish", "French"])
il = get_infolingo(language_choice.lower())

text = st.text_area("Paste in text:", height=300)

if text:
    words = tokenize_text(text)
    method_choice = st.selectbox(
        "Vocab selection method:", ["Cross-Entropy", "Frequent", "Random", "KL-Divergence"]
    )
    max_slider = min(len(words), 100)
    vocab_size = st.slider("Select number of new vocabulary:", 0, max_slider)
    get_vocab_btn = st.button("Get Vocab!")
    if get_vocab_btn:
        vocab = il.pick_vocab(text, n=vocab_size, method=method_choice.lower())
        st.markdown("**Your Vocab List**")
        for v in vocab:
            st.markdown(f"- {v}")
