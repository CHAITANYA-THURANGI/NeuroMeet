"""Evaluation benchmark script for NeuroMeet."""

from __future__ import annotations
import argparse
import time
from typing import Dict, List
import numpy as np

from src.datasets.generator import generate_meeting_scenarios
from src.evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    diarization_error_rate,
    evaluate_action_items,
)
from src.pipeline.omni_meeting import OmniMeetingPipeline


def evaluate_system() -> None:
    print("=" * 78)
    print(" NeuroMeet: Deep Learning Meeting Assistant Benchmark Suite")
    print("=" * 78)

    pipeline = OmniMeetingPipeline()
    scenarios = generate_meeting_scenarios()

    r1_scores, r2_scores, rl_scores = [], [], []
    bleu_scores = []
    f1_scores = []
    latencies = []

    print(f"\nEvaluating across {len(scenarios)} enterprise meeting benchmarks:\n")
    print(f"{'Scenario Name':<35} | {'ROUGE-1':<8} | {'ROUGE-L':<8} | {'BLEU':<6} | {'Action F1':<9} | {'Latency':<8}")
    print("-" * 84)

    for sc_id, sc in scenarios.items():
        t0 = time.perf_counter()
        result = pipeline.process_transcript(sc.turns, title=sc.title)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Summarization metrics
        rouge = compute_rouge(sc.ground_truth_summary, result.minutes.executive_summary)
        bleu = compute_bleu(sc.ground_truth_summary, result.minutes.executive_summary)

        # Action item metrics
        pred_actions = [{"task": a.task, "assignee": a.assignee} for a in result.action_items]
        act_metrics = evaluate_action_items(sc.ground_truth_action_items, pred_actions)

        r1_scores.append(rouge["rouge1"])
        r2_scores.append(rouge["rouge2"])
        rl_scores.append(rouge["rougeL"])
        bleu_scores.append(bleu)
        f1_scores.append(act_metrics["f1"])
        latencies.append(latency_ms)

        print(f"{sc.title[:34]:<35} | {rouge['rouge1']:<8.2f} | {rouge['rougeL']:<8.2f} | {bleu:<6.2f} | {act_metrics['f1']:<9.1f} | {latency_ms:<6.1f}ms")

    print("-" * 84)
    print(f"{'MACRO AVERAGE':<35} | {np.mean(r1_scores):<8.2f} | {np.mean(rl_scores):<8.2f} | {np.mean(bleu_scores):<6.2f} | {np.mean(f1_scores):<9.1f} | {np.mean(latencies):<6.1f}ms")
    print("=" * 78)


if __name__ == "__main__":
    evaluate_system()
