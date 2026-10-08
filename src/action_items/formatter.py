"""Export formatters for meeting action items."""

from __future__ import annotations
from typing import Any, Dict, List
from .extractor import ActionItem


def format_action_items_markdown(items: List[ActionItem]) -> str:
    """Formats list of ActionItem dataclasses into a GitHub-flavored Markdown task list."""
    if not items:
        return "No action items identified for this session."

    lines = ["### Action Items\n"]
    for i, it in enumerate(items, 1):
        prio_badge = f"**[{it.priority.upper()}]**"
        lines.append(f"- [ ] {prio_badge} **{it.task}** — Assigned to *{it.assignee}* (Due: {it.deadline})")
    return "\n".join(lines)


def format_action_items_jira(items: List[ActionItem], project_key: str = "MEET") -> List[Dict[str, Any]]:
    """Formats action items into standard Atlassian Jira issue creation payloads."""
    jira_issues = []
    prio_map = {
        "urgent": "Highest",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }

    for item in items:
        jira_issues.append({
            "fields": {
                "project": {"key": project_key},
                "summary": item.task,
                "description": f"Extracted automatically by NeuroMeet AI.\nSource: \"{item.source_utterance}\"\nDue: {item.deadline}",
                "issuetype": {"name": "Task"},
                "priority": {"name": prio_map.get(item.priority.lower(), "Medium")},
                "assignee": {"name": item.assignee.lower().replace(" ", ".")},
            }
        })
    return jira_issues
