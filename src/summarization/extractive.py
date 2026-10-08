"""Graph-based TextRank & Centrality Extractive Summarizer for meetings."""

from __future__ import annotations
import math
import re
from typing import List, Set, Tuple
import numpy as np


# English conversational stopwords
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "did", "do", "does", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves", "yeah", "okay",
    "um", "uh", "like", "right", "know", "think"
}

# Hindi conversational stopwords & particles (Devanagari & Romanized Hinglish)
HINDI_STOPWORDS: Set[str] = {
    "है", "हैं", "का", "की", "के", "में", "और", "तो", "भी", "से", "को", "पर", "यह", "वह",
    "था", "थी", "थे", "गया", "कर", "दिया", "लिया", "होता", "सकता", "चाहिए", "नहीं", "मत",
    "hai", "hain", "kaa", "kii", "ke", "mein", "aur", "toh", "bhi", "se", "ko", "par",
    "tha", "thi", "the", "kare", "karna", "nahi"
}

# Telugu conversational stopwords & particles (Telugu script & Romanized Tenglish)
TELUGU_STOPWORDS: Set[str] = {
    "మరియు", "కానీ", "లో", "యొక్క", "అని", "తో", "నుండి", "కోసం", "ద్వారా", "వద్ద",
    "ఉంది", "ఉన్నారు", "చేయండి", "చేసారు", "లేదు", "కాదు", "అవును", "ఇది", "అది",
    "mariyu", "kaani", "kani", "lo", "yokka", "ani", "tho", "nundi", "kosam",
    "undi", "unnaru", "ledu", "kadu", "idi", "adi", "inka"
}

ALL_STOPWORDS: Set[str] = STOPWORDS | HINDI_STOPWORDS | TELUGU_STOPWORDS


def tokenize_words(text: str) -> List[str]:
    """Tokenizes text into lowercase words across English, Hindi (Devanagari/Romanized), and Telugu."""
    words = re.findall(r"\b[\w\-]+\b", text.lower(), flags=re.UNICODE)
    return [w for w in words if w not in ALL_STOPWORDS and len(w) > 1]


class TextRankExtractiveSummarizer:
    """Extracts the most informative sentences using PageRank on the semantic graph."""

    def __init__(self, damping: float = 0.85, max_iter: int = 50, tol: float = 1e-4) -> None:
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def summarize(self, utterances: List[str], top_n: int = 4) -> List[str]:
        """Selects top_n most central utterances from the meeting transcript."""
        n = len(utterances)
        if n == 0:
            return []
        if n <= top_n:
            return utterances

        # Tokenize utterances
        tokenized = [tokenize_words(u) for u in utterances]

        # Compute pairwise BM25 / Jaccard similarity matrix
        sim_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            set_i = set(tokenized[i])
            if not set_i:
                continue
            for j in range(i + 1, n):
                set_j = set(tokenized[j])
                if not set_j:
                    continue
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                if union > 0:
                    weight = intersection / math.sqrt(len(set_i) * len(set_j))
                    sim_matrix[i, j] = weight
                    sim_matrix[j, i] = weight

        # Row normalization for transition matrix
        row_sums = sim_matrix.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        transition = sim_matrix / row_sums[:, np.newaxis]

        # PageRank power iteration
        scores = np.ones(n, dtype=np.float32) / n
        for _ in range(self.max_iter):
            prev_scores = scores.copy()
            scores = (1.0 - self.damping) / n + self.damping * np.dot(transition.T, prev_scores)
            if np.linalg.norm(scores - prev_scores) < self.tol:
                break

        # Select top_n indices while preserving chronological order
        top_indices = np.argsort(scores)[::-1][:top_n]
        top_indices = sorted(top_indices)

        return [utterances[idx] for idx in top_indices]
