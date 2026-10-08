"""Deep Neural & Syntactic Action Item Extractor."""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Dict, List, Optional
import torch
from ..models.action_extractor import ActionItemClassifier


ACTION_PATTERNS = [
    r"\b(?:i will|i'll|i can)\s+([^,.;]+)",
    r"\b(?:let's|we need to|we should|we have to)\s+([^,.;]+)",
    r"\b([a-zA-Z]+)\s+(?:will|should|to take care of|needs to)\s+([^,.;]+)",
    r"\baction item[:\s]+([^,.;]+)",
    r"\bassigned to\s+([a-zA-Z]+)[:\s]+([^,.;]+)",
]

DEADLINE_PATTERNS = [
    r"\b(?:by|before|due)\s+([a-zA-Z0-9\s]+?(?:eod|friday|monday|tuesday|wednesday|thursday|tomorrow|next week|end of week|end of month|q[1-4]))",
    r"\b(?:by|before)\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)",
    r"\b(?:by)\s+([a-zA-Z]+\s+[0-9]{1,2})",
]

URGENT_WORDS = {"urgent", "asap", "blocker", "critical", "immediately", "high priority", "p0", "p1"}


@dataclass
class ActionItem:
    """Structured actionable commitment from a meeting."""
    task: str
    assignee: str
    deadline: str
    priority: str
    confidence: float
    source_utterance: str


def simple_hash_tokens(text: str, vocab_size: int = 4000) -> torch.Tensor:
    """Converts text tokens to tensor IDs."""
    words = text.split()
    ids = []
    for w in words:
        clean = w.lower().strip()
        val = 1 + (abs(hash(clean)) % (vocab_size - 1)) if clean else 0
        ids.append(val)
    if not ids:
        ids = [0]
    return torch.tensor(ids, dtype=torch.long)


class DeepActionExtractor:
    """Neural Sequence Tagger + Syntactic Parser for extracting actionable commitments."""

    def __init__(
        self,
        classifier: Optional[ActionItemClassifier] = None,
        vocab_size: int = 4000,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.classifier = classifier.to(self.device).eval() if classifier else None
        self.vocab_size = vocab_size

    def extract_from_utterance(
        self,
        utterance: str,
        current_speaker: str = "Unassigned",
    ) -> Optional[ActionItem]:
        """Extracts an ActionItem if the utterance contains an explicit or implicit task commitment."""
        clean = utterance.strip()
        lower = clean.lower()

        # Neural evaluation if model is loaded
        neural_pred = None
        if self.classifier is not None:
            input_ids = simple_hash_tokens(clean, self.vocab_size).to(self.device)
            words = clean.split()
            neural_pred = self.classifier.extract_action(words, input_ids, threshold=0.40)

        # Syntactic extraction
        matched_task = ""
        matched_assignee = current_speaker
        matched_deadline = "Next sprint"
        matched_priority = "medium"

        # Check for deadline cues
        for d_pat in DEADLINE_PATTERNS:
            d_match = re.search(d_pat, lower)
            if d_match:
                matched_deadline = d_match.group(1).strip().capitalize()
                break

        # Check for urgency cues
        if any(w in lower for w in URGENT_WORDS):
            matched_priority = "urgent"
        elif any(w in lower for w in ["important", "soon", "priority", "needed"]):
            matched_priority = "high"

        # Check for action task patterns
        for pat in ACTION_PATTERNS:
            match = re.search(pat, clean, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    matched_task = groups[0].strip()
                elif len(groups) == 2:
                    potential_who = groups[0].strip()
                    if potential_who.lower() not in {"we", "let's", "i"}:
                        matched_assignee = potential_who
                    matched_task = groups[1].strip()
                break

        if not matched_task and neural_pred is not None:
            matched_task = neural_pred["task"]
            if neural_pred["assignee"] != "Unassigned":
                matched_assignee = neural_pred["assignee"]
            if neural_pred["deadline"] != "Next sync":
                matched_deadline = neural_pred["deadline"]
            matched_priority = neural_pred["priority"]

        if matched_task and len(matched_task) > 5:
            conf = neural_pred["confidence"] if neural_pred else 0.88
            return ActionItem(
                task=matched_task,
                assignee=matched_assignee,
                deadline=matched_deadline,
                priority=matched_priority,
                confidence=conf,
                source_utterance=clean,
            )

        return None

    def extract_all(
        self,
        turns: List[Dict[str, str]],
    ) -> List[ActionItem]:
        """Extracts all unique action items across meeting dialogue turns."""
        items: List[ActionItem] = []
        seen_tasks = set()

        for turn in turns:
            speaker = turn.get("speaker", "Unassigned")
            text = turn.get("text", "")
            action = self.extract_from_utterance(text, current_speaker=speaker)
            if action:
                task_norm = action.task.lower().strip()
                if task_norm not in seen_tasks:
                    seen_tasks.add(task_norm)
                    items.append(action)

        return items
