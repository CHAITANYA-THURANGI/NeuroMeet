"""Comprehensive evaluation metrics for NeuroMeet.
Implements WER, CER, DER, ROUGE-1/2/L, BLEU, and Action Item F1.
"""

from __future__ import annotations
import math
import re
from typing import Any, Dict, List, Set, Tuple
import numpy as np


def levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> int:
    """Computes minimum edit distance (Insertions, Deletions, Substitutions)."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[m][n]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate: (S + D + I) / N_ref."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return round(float(dist / len(ref_words)), 4)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Computes Character Error Rate: (S + D + I) / N_ref_chars."""
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return round(float(dist / len(ref_chars)), 4)


def diarization_error_rate(
    ref_turns: List[Dict[str, Any]],
    hyp_turns: List[Dict[str, Any]],
    total_duration_sec: float,
) -> Dict[str, float]:
    """Computes Diarization Error Rate (DER):
    DER = (Missed Speech + False Alarm + Speaker Error) / Total Speech Time
    """
    if total_duration_sec <= 0.0:
        return {"der": 0.0, "missed": 0.0, "false_alarm": 0.0, "speaker_error": 0.0}

    # Discretize timeline into 100ms frames
    frame_step = 0.1
    n_frames = int(math.ceil(total_duration_sec / frame_step))
    ref_frames = [set() for _ in range(n_frames)]
    hyp_frames = [set() for _ in range(n_frames)]

    for t in ref_turns:
        s = int(t["start_sec"] / frame_step)
        e = int(t["end_sec"] / frame_step)
        spk = t["speaker_id"]
        for f in range(s, min(e, n_frames)):
            ref_frames[f].add(spk)

    for t in hyp_turns:
        s = int(t["start_sec"] / frame_step)
        e = int(t["end_sec"] / frame_step)
        spk = t["speaker_id"]
        for f in range(s, min(e, n_frames)):
            hyp_frames[f].add(spk)

    missed_frames = 0
    false_alarm_frames = 0
    speaker_error_frames = 0
    total_ref_speech_frames = 0

    for f in range(n_frames):
        r_spks = ref_frames[f]
        h_spks = hyp_frames[f]

        total_ref_speech_frames += len(r_spks)

        if len(r_spks) > len(h_spks):
            missed_frames += (len(r_spks) - len(h_spks))
        elif len(h_spks) > len(r_spks):
            false_alarm_frames += (len(h_spks) - len(r_spks))

        # Check speaker mapping disagreement
        if len(r_spks) > 0 and len(h_spks) > 0:
            if not r_spks.intersection(h_spks):
                speaker_error_frames += min(len(r_spks), len(h_spks))

    ref_denom = max(1, total_ref_speech_frames)
    der = (missed_frames + false_alarm_frames + speaker_error_frames) / ref_denom

    return {
        "der": round(min(1.0, der), 4),
        "missed": round(missed_frames / ref_denom, 4),
        "false_alarm": round(false_alarm_frames / ref_denom, 4),
        "speaker_error": round(speaker_error_frames / ref_denom, 4),
    }


def get_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # ROUGE-1
    ref_1 = get_ngrams(ref_tokens, 1)
    hyp_1 = get_ngrams(hyp_tokens, 1)
    overlap_1 = sum(min(count, hyp_1.get(gram, 0)) for gram, count in ref_1.items())
    prec_1 = overlap_1 / len(hyp_tokens)
    rec_1 = overlap_1 / len(ref_tokens)
    f1_1 = (2 * prec_1 * rec_1) / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0

    # ROUGE-2
    ref_2 = get_ngrams(ref_tokens, 2)
    hyp_2 = get_ngrams(hyp_tokens, 2)
    overlap_2 = sum(min(count, hyp_2.get(gram, 0)) for gram, count in ref_2.items())
    prec_2 = overlap_2 / max(1, len(hyp_tokens) - 1)
    rec_2 = overlap_2 / max(1, len(ref_tokens) - 1)
    f1_2 = (2 * prec_2 * rec_2) / (prec_2 + rec_2) if (prec_2 + rec_2) > 0 else 0.0

    # ROUGE-L (Longest Common Subsequence)
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec_l = lcs / n
    rec_l = lcs / m
    f1_l = (2 * prec_l * rec_l) / (prec_l + rec_l) if (prec_l + rec_l) > 0 else 0.0

    return {
        "rouge1": round(f1_1 * 100.0, 2),
        "rouge2": round(f1_2 * 100.0, 2),
        "rougeL": round(f1_l * 100.0, 2),
    }


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Computes BLEU-4 score with brevity penalty."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(ref_tokens, n)
        hyp_ngrams = get_ngrams(hyp_tokens, n)
        total_hyp = max(1, len(hyp_tokens) - n + 1)
        overlap = sum(min(count, hyp_ngrams.get(gram, 0)) for gram, count in ref_ngrams.items())
        precisions.append((overlap + 1e-4) / total_hyp)

    log_sum = sum(math.log(p) for p in precisions) / max_n
    geo_mean = math.exp(log_sum)

    # Brevity penalty
    c = len(hyp_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1.0 - (r / max(1, c)))

    return round(bp * geo_mean * 100.0, 2)


def evaluate_action_items(
    ref_items: List[Dict[str, str]],
    hyp_items: List[Dict[str, str]],
) -> Dict[str, float]:
    """Computes Precision, Recall, and F1 on extracted action items based on task overlap."""
    if not ref_items and not hyp_items:
        return {"precision": 100.0, "recall": 100.0, "f1": 100.0}
    if not ref_items or not hyp_items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched = 0
    for h in hyp_items:
        h_words = set(h.get("task", "").lower().split())
        for r in ref_items:
            r_words = set(r.get("task", "").lower().split())
            if len(h_words.intersection(r_words)) >= max(1, min(len(h_words), len(r_words)) // 2):
                matched += 1
                break

    prec = matched / len(hyp_items)
    rec = matched / len(ref_items)
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(prec * 100.0, 2),
        "recall": round(rec * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
    }
