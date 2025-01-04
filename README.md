<p align="center">
  <img src="https://github.com/aliceheiman/infolingo/blob/main/assets/logo.png" alt="Infolingo logo" width="200px" />
</p>

# 🌏 Infolingo – Efficient Vocabulary Selection for Foreign-Language Learning

![Python](https://img.shields.io/badge/python-3.x-blue.svg) [![MIT license](https://img.shields.io/badge/License-MIT-green.svg)](https://lbesson.mit-license.org/)

Infolingo uses probability to pick the best words to learn next to improve understanding of a foreign language text.

Check out the live [demo](http://infolingo.streamlit.app/).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install infolingo.

```shell
pip install infolingo
```

### (Optional) Streamlit Demo GUI

```shell
# download repo
git clone
python -m venv .venv
source .venv/bin/activate

# start demo GUI
cd streamlit_demo
pip install -r requirements.txt
streamlit run app.py
```

You should then see a locally hosted website like this:

<p align="center">
  <img src="https://github.com/aliceheiman/infolingo/blob/main/assets/infolingo-front.png" alt="Infolingo demo" width="500px" />
</p>

## Usage

Quickstart using English as the default language and Cross-Entropy as the default vocabulary picking function. 

```python
from infolingo import Infolingo

il = Infolingo(language="english")
vocab = il.pick_vocab("The quick brown fox jumps over the lazy dog", n=2)
print(vocab) # prints ["jumps", "fox"]
```

### Supported Languages

```python
Infolingo(language="english")
Infolingo(language="spanish")
Infolingo(language="french")
```

### Custom Corpus

Format your corpus file as a **CSV** with fields **word,frequency** and double quote (") as a delimiter character.

```python
Infolingo(language="language", custom_vocab_file="path/to/custom/corpus")
```

## Vocabulary Picking Functions

We evaluated four vocabulary-picking functions. The results indicate that cross-entropy and KL-divergence are most effective for language comprehension.

### Cross-Entropy

Selects the top n vocabulary that decreases cross-entropy for the text the most.

```python
il = Infolingo()
vocab = il.pick_vocab(text, n=3, method="cross-entropy")
```

### KL-Divergence

Select the top n vocabulary that decreases [KL-Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) for the text the most.

```python
from infolingo import Infolingo
il = Infolingo()
vocab = il.pick_vocab(text, n=3, method="kl-divergence")
```

### Frequent

Select the top n most frequent words in the text.

```python
from infolingo import Infolingo
il = Infolingo()
vocab = il.pick_vocab(text, n=3, method="frequent")
```

### Random

Select n random words from the text.

```python
from infolingo import Infolingo
il = Infolingo()
vocab = il.pick_vocab(text, n=3, method="random")
```

## Default Corpora

The default corpora used are listed below:

- English: [Brown Corpus](https://www.nltk.org/book/ch02.html)
- Spanish: [Wortschatz Leipzig. spa_news_2023_300K-words](https://corpora.wortschatz-leipzig.de/en?corpusId=spa_news_2023)
- French: [Wortschatz Leipzig. fra_news_2023_300K-words](https://corpora.wortschatz-leipzig.de/en?corpusId=fra_news_2023)

To use your corpus (alternative corpus to the ones above or to support a new language), see "Custom Corpus" above.

## Contributing

Any contributions you make are **greatly appreciated**. 

If you have a suggestion to improve this, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement."
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Changelog

#### 1.0.1
Update README.md and links.

### 1.0.0
Initial infolingo PyPi submission. This version supports cross-entropy, kl-divergence, frequent, and random vocabulary picking functions.
It contains a streamlit demo for testing.

## License

[MIT](https://choosealicense.com/licenses/mit/)