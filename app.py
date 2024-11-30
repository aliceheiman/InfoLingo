import streamlit as st
from utils import (
    get_corpus_word_probs,
    tokenize_text,
    pick_baseline_frequent,
    pick_baseline_random,
    pick_entropy,
    pick_kl_divergence,
    get_word_stats,
    pick_top_n_bigrams,
    pick_top_n_trigrams,
    pick_top_n_zipf_deviation,
)


### Setup
def get_top_n_vocab(text, method, n, prior_vocab=[]):
    words = tokenize_text(text)
    words = [word for word in words if word not in prior_vocab]  # prior knowledge
    word_counts, word_probs = get_word_stats(words)

    vocab = []

    # get the next vocab
    if method == "Frequent":
        vocab = pick_baseline_frequent(words, word_counts, word_probs, n)
    if method == "Random":
        vocab = pick_baseline_random(words, word_counts, word_probs, n)
    if method == "Entropy":
        vocab = pick_entropy(words, word_counts, word_probs, corpus_word_probs, n)
    if method == "KL-Divergence":
        vocab = pick_kl_divergence(words, word_counts, word_probs, corpus_word_probs, n)
    return vocab


st.image("assets/banner.png")
st.title("👋🌎 Welcome to InfoLingo!")
st.info(
    "This tool uses information theory to select the most optimal words to learn given a piece of text you want to understand. To use this app, select your target language, paste in your text, and select your target vocabulary!"
)

language = st.radio("Pick a Language", ["English", "Spanish", "French"])
corpus_word_probs = get_corpus_word_probs(language)

text = st.text_area("Paste in text:", height=300)
words = tokenize_text(text)

if text:
    method = st.selectbox(
        "Vocab selection method:", ["Entropy", "Frequent", "Random", "KL-Divergence"]
    )
    max_slider = min(len(words), 100)
    vocab_size = st.slider("Select number of new vocabulary:", 0, max_slider)
    get_vocab_btn = st.button("Get Vocab!")
    if get_vocab_btn:
        vocab = get_top_n_vocab(text, method, vocab_size)
        st.markdown("**Your Vocab List**")
        for v in vocab:
            st.markdown(f"- {v}")

    ngram_n = st.slider("Select number of bigrams/trigrams:", 0, 20)
    col1, col2 = st.columns(2)
    with col1:
        get_bigrams_btn = st.button("Get Bigrams!")
        if get_bigrams_btn:
            bigrams = pick_top_n_bigrams(text, ngram_n)
            st.markdown("**Your Bigram List**")
            for bigram in bigrams:
                st.markdown(f"- {bigram}")

    with col2:
        get_trigrams_btn = st.button("Get Trigrams!")
        if get_trigrams_btn:
            trigrams = pick_top_n_trigrams(text, ngram_n)
            st.markdown("**Your Trigram List**")
            for trigram in trigrams:
                st.markdown(f"- {trigram}")


# st.subheader("Author Analysis")
