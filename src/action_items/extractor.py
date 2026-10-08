"""Deep Neural & Syntactic Action Item Extractor."""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Dict, List, Optional
import torch
from ..models.action_extractor import ActionItemClassifier


ACTION_PATTERNS = [
    # English patterns
    r"\b(?:i will|i'll|i can|i am going to|i'm going to|i'll handle|i'll take care of|i'll take the lead on|i'll follow up on|i am on|i'm on)\s+([^,.;]+)",
    r"\b(?:let's|we need to|we should|we have to|we ought to|we must|please make sure to|please ensure)\s+([^,.;]+)",
    r"\b([a-zA-Z\u0900-\u097F\u0C00-\u0C7F]+)\s+(?:will|should|to take care of|needs to|is going to|to handle|to lead|to follow up on)\s+([^,.;]+)",
    r"\baction item[:\s]+([^,.;]+)",
    r"\b(?:todo|task)[:\s]+([^,.;]+)",
    r"\bassigned to\s+([a-zA-Z\u0900-\u097F\u0C00-\u0C7F]+)[:\s]+([^,.;]+)",
    # Hindi patterns (Devanagari & Hinglish)
    r"\b(?:main|mai|hum|aap)\s+([^,.;।]+?(?:\bkarunga\b|\bkar dungi\b|\bkarenge\b|\bkarna hai\b|\bdekhunga\b|\bdekhenge\b|\bkar doongi\b))",
    r"(?:मैं|हम|आप)\s+([^।\n,;]+?(?:करूँगी|करूंगी|करूँगा|करूंगा|करेंगे|करेगा|करेगी|कर देना|करना है|दूँगी|दूंगी|दूँगा|दूंगा|देखेंगे|देखूँगी|देखूँगा))",
    r"\b([a-zA-Z\u0900-\u097F]+)\s+(?:ye karega|karega|karenge|karna padega|करेगा|करेगी|करेंगे)",
    # Telugu patterns (Telugu script & Tenglish)
    r"\b(?:nenu|manam|meeru)\s+([^,.;।]+?(?:\bchestanu\b|\bpampistanu\b|\bcheyali\b|\bcheddam\b|\bchustanu\b|\bchudali\b))",
    r"(?:నేను|మనం|మీరు)\s+([^।\n.;,]+?(?:చేస్తాను|చేయాలి|చేద్దాం|చూస్తాను|చూడాలి|పంపిస్తాను|తీసుకుంటాను|ఇస్తాను|రాస్తాను|చేస్తాం|పంపిస్తాం))",
    r"\b([a-zA-Z\u0C00-\u0C7F]+)\s+(?:chestadu|chestaru|cheyali|chustaru|చేస్తారు|చేస్తుంది|చేయాలి)",
]

DEADLINE_PATTERNS = [
    # English deadlines
    r"\b(?:by|before|due)\s+([a-zA-Z0-9\s]+?(?:eod|cob|cop|friday|monday|tuesday|wednesday|thursday|saturday|sunday|tomorrow|tomorrow\s+morning|tomorrow\s+afternoon|tomorrow\s+evening|tomorrow\s+eod|next week|this week|end of week|end of month|end of sprint|next sprint|q[1-4]))",
    r"\b(?:by|before)\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)",
    r"\b(?:by)\s+([a-zA-Z]+\s+[0-9]{1,2}(?:st|nd|rd|th)?)",
    r"\b(?:in|within)\s+([0-9]+\s+(?:days|hours|weeks))",
    r"\b(?:deadline is|due on)\s+([a-zA-Z0-9\s]+)",
    # Hindi deadlines (कल शाम तक, कल तक, शुक्रवार तक, सोमवार तक, अगले हफ्ते, kal tak, shukrawar tak, kal shaam tak)
    r"((?:कल|आज|शुक्रवार|सोमवार|मंगलवार|बुधवार|गुरुवार|शनिवार|रविवार|अगले\s+हफ्ते)(?:\s+(?:सुबह|दोपहर|शाम|रात|ईओडी))?\s*तक)",
    r"([a-zA-Z\u0900-\u097F0-9\s]+?(?:kal tak|shukrawar tak|somwar tak|agale hafte|kal shaam tak|kal sham tak))",
    # Telugu deadlines (రేపు, శుక్రవారం లోగా, శుక్రవారం లోపల, సోమవారం లోగా, repu, repu morning, shukravaram loga)
    r"((?:రేపు|ఈరోజు|శుక్రవారం|సోమవారం|మంగళవారం|బుधవారం|గురువారం|శనివారం|ఆదివారం|వచ్చే\s+వారం)(?:\s+(?:ఉదయం|సాయంత్రం|మధ్యాహ్నం))?\s*(?:లోగా|లోపల|వరకు))",
    r"([a-zA-Z\u0C00-\u0C7F0-9\s]+?(?:repu\s*morning|repu\s*afternoon|repu|shukravaram loga|shukravaram lopala|somavaram loga|vache vaaram|repu lopu))",
]

URGENT_WORDS = {
    # English
    "urgent", "asap", "blocker", "critical", "immediately", "high priority", "p0", "p1",
    # Hindi
    "jaldi", "turant", "zaroori", "bohot zaroori", "जल्दी", "तुरंत", "ज़रूरी", "जरूरी",
    # Telugu
    "ventane", "tvaraga", "urgent ga", "chala important", "వెంటనే", "త్వరగా", "ముఖ్యం"
}


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
            d_match = re.search(d_pat, clean, re.IGNORECASE)
            if d_match:
                deadline_str = d_match.group(1).strip()
                matched_deadline = deadline_str.capitalize() if deadline_str.isascii() else deadline_str
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
