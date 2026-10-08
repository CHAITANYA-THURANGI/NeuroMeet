"""Meeting Dataset loader, tensor batching, and PyTorch DataLoader."""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from .generator import MeetingScenario, generate_meeting_scenarios


def simple_hash_tokens(text: str, vocab_size: int = 4000) -> List[int]:
    words = text.split()
    ids = []
    for w in words:
        clean = w.lower().strip()
        val = 1 + (abs(hash(clean)) % (vocab_size - 1)) if clean else 0
        ids.append(val)
    return ids or [0]


class MeetingDataset(Dataset):
    """PyTorch Dataset yielding formatted tensors for meeting summarization and action items."""

    def __init__(
        self,
        scenarios: List[MeetingScenario],
        vocab_size: int = 4000,
        max_utts: int = 32,
        words_per_utt: int = 32,
        max_target_len: int = 64,
    ) -> None:
        self.scenarios = scenarios
        self.vocab_size = vocab_size
        self.max_utts = max_utts
        self.words_per_utt = words_per_utt
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sc = self.scenarios[idx]

        # Meeting input tensor: [num_utts, words_per_utt]
        meeting_t = torch.zeros((self.max_utts, self.words_per_utt), dtype=torch.long)
        num_turns = min(len(sc.turns), self.max_utts)

        for u_idx in range(num_turns):
            text = sc.turns[u_idx]["text"]
            ids = simple_hash_tokens(text, self.vocab_size)[: self.words_per_utt]
            meeting_t[u_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        # Target summary tensor: [max_target_len]
        target_t = torch.zeros(self.max_target_len, dtype=torch.long)
        tgt_ids = simple_hash_tokens(sc.ground_truth_summary, self.vocab_size)[: self.max_target_len]
        target_t[: len(tgt_ids)] = torch.tensor(tgt_ids, dtype=torch.long)

        return {
            "scenario_id": sc.scenario_id,
            "title": sc.title,
            "meeting_tokens": meeting_t,
            "target_tokens": target_t,
            "num_turns": num_turns,
        }


def create_dataloader(
    scenarios: Optional[List[MeetingScenario]] = None,
    batch_size: int = 2,
    shuffle: bool = True,
    vocab_size: int = 4000,
) -> DataLoader:
    """Creates a PyTorch DataLoader for meeting scenarios."""
    if scenarios is None:
        scenarios_dict = generate_meeting_scenarios()
        scenarios = list(scenarios_dict.values())

    dataset = MeetingDataset(scenarios, vocab_size=vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
