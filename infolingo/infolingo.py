# infolingo.py
import math
from collections import Counter
from random import sample

from .main import get_word_stats, tokenize_text, get_word_probs, load_word_counts_from_file, \
    load_word_counts_from_language


class Infolingo:
    def __init__(self, language: str = "english", custom_vocab_file: str = None):
        language = language.lower()
        self.language = language
        self.custom_vocab_file = custom_vocab_file

        if custom_vocab_file:
            self.corpus_word_counts = load_word_counts_from_file(custom_vocab_file)
        elif language:
            self.corpus_word_counts = load_word_counts_from_language(language)

        self.corpus_probs = get_word_probs(self.corpus_word_counts)

    def pick_cross_entropy(self, words: list[str], n: int) -> list[str]:
        """
        Select the top n words that reduce cross-entropy the most.
        :param words: list of candidate words.
        :param n: number of words to select.
        :return: list of top n words using cross-entropy.
        """

        def _compute_entropy(w_counts, w_probs):
            entropy = 0
            for w in w_counts:
                entropy += math.log2(1 / self.corpus_probs.get(w, 0.00001)) * w_probs[w]
            return entropy

        # baseline entropy of text
        word_counts, word_probs = get_word_stats(words)
        baseline_entropy = _compute_entropy(word_counts, word_probs)
        deltas = []

        # remove each word in turn
        for word in word_counts:
            candidate_words = [w for w in words if w != word]
            candidate_word_counts, candidate_probs = get_word_stats(candidate_words)
            candidate_entropy = _compute_entropy(candidate_word_counts, candidate_probs)

            delta = baseline_entropy - candidate_entropy
            deltas.append((word, delta))

        # Pick the top words
        deltas = sorted(deltas, key=lambda x: x[1], reverse=True)
        picks = [w[0] for w in deltas[:n]]
        return picks

    def pick_random(self, words: list[str], n: int) -> list[str]:
        """
        Select random n words.
        :param words: list of candidate words.
        :param n: number of words to select.
        :return: list of top n words using random selection.
        """
        word_counts = dict(Counter(words))
        picks = sample(list(word_counts.keys()), n)
        return picks

    def pick_frequent(self, words: list[str], n: int) -> list[str]:
        """
        Select the most frequent n words.
        :param words: list of candidate words.
        :param n: number of words to select.
        :return: list of top n words using frequent selection.
        """
        word_counts = Counter(words)
        picks = word_counts.most_common(n)
        picks = [p[0] for p in picks]
        return picks

    def pick_kl_divergence(self, words: list[str], n: int) -> list[str]:
        """
        Select top n words using kl divergence.
        :param words: list of candidate words.
        :param n: number of words to select.
        :return: list of top n words using kl divergence selection.
        """
        word_counts, word_probs = get_word_stats(words)
        all_words = set(self.corpus_probs.keys()).union(set(words))

        smoothing = 0.00001
        p_smoothed = {w: self.corpus_probs.get(w, 0) + smoothing for w in all_words}
        q_smoothed = {w: word_probs.get(w, 0) + smoothing for w in all_words}

        p_total = sum(p_smoothed.values())
        q_total = sum(q_smoothed.values())

        p_normalized = {w: p / p_total for w, p in p_smoothed.items()}
        q_normalized = {w: q / q_total for w, q in q_smoothed.items()}

        kl_divergence = 0.0
        contributions = {}
        for w in all_words:
            q = q_normalized[w]
            p = p_normalized[w]
            if q > 0:  # Skip if Q(w) is 0 (log(0) is undefined)
                contribution = q * math.log(q / p)
                kl_divergence += contribution
                contributions[w] = contribution

        sorted_words = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        picks = [w[0] for w in sorted_words[:n]]
        return picks

    def pick_vocab(self, text: str, n: int, method: str = "cross-entropy") -> list[str]:
        """
        Select best vocabulary to learn from text.
        :param text: input text.
        :param n: number of words to select.
        :param method: method for selecting vocabulary.
            Supported methods: "cross-entropy", "random", "frequent"
            "kl-divergence"
        :return: list of top n words to learn using the method.
        """
        method = method.lower()
        words = tokenize_text(text)
        if method == "cross-entropy":
            return self.pick_cross_entropy(words, n)
        elif method == "random":
            return self.pick_random(words, n)
        elif method == "frequent":
            return self.pick_frequent(words, n)
        elif method == "kl-divergence":
            return self.pick_kl_divergence(words, n)
        else:
            raise ValueError("Method not supported.")
