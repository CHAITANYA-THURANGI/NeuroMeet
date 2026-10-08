"""Interactive Command-Line Demonstration of NeuroMeet."""

from __future__ import annotations
import argparse
from pathlib import Path
from src.pipeline.omni_meeting import OmniMeetingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroMeet CLI Meeting Intelligence")
    parser.add_argument("--audio", type=str, default=None, help="Path to WAV audio file")
    parser.add_argument("--transcript", type=str, default=None, help="Path to text transcript file")
    parser.add_argument("--scenario", type=str, default="sprint_planning", help="Sample scenario name")
    args = parser.parse_args()

    pipeline = OmniMeetingPipeline()

    if args.audio:
        print(f"\nProcessing audio file: {args.audio}...")
        result = pipeline.process_audio(args.audio, title=Path(args.audio).stem)
    elif args.transcript:
        print(f"\nProcessing transcript file: {args.transcript}...")
        text = Path(args.transcript).read_text(encoding="utf-8")
        result = pipeline.process_transcript(text, title=Path(args.transcript).stem)
    else:
        from src.datasets.generator import generate_meeting_scenarios
        scenarios = generate_meeting_scenarios()
        sc = scenarios.get(args.scenario, list(scenarios.values())[0])
        print(f"\nProcessing pre-loaded scenario: {sc.title}...")
        result = pipeline.process_transcript(sc.turns, title=sc.title)

    print("\n" + "=" * 70)
    print(f"  {result.title}")
    print(f"  Health Score: {result.health.overall_score}/100 ({result.health.grade}) — {result.health.category}")
    print("=" * 70)

    print("\n[EXECUTIVE SUMMARY]")
    print(result.minutes.executive_summary)

    print("\n[KEY DECISIONS]")
    for dec in result.minutes.key_decisions:
        print(f"  * {dec}")

    print("\n[ACTION ITEMS]")
    for item in result.action_items:
        print(f"  [ ] [{item.priority.upper()}] {item.task} — Assigned to: {item.assignee} (Due: {item.deadline})")

    print("\n[PARTICIPATION & DOMINANCE]")
    print(f"  Total Turns: {result.participation.total_turns} | Dominant Speaker: {result.participation.dominant_speaker}")
    for spk, pct in result.participation.talk_time_percentages.items():
        print(f"  * {spk:<25}: {pct}%")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
