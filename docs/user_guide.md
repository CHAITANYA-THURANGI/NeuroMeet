# NeuroMeet User Guide & Manual

Welcome to **NeuroMeet AI**. This guide walks through the primary user flows for the Web Studio and Browser Extension.

---

## 1. Web Intelligence Studio (`http://127.0.0.1:8000`)

### Quick Scenarios
At the top of the interface, four one-click enterprise scenarios are provided:
- **⚡ Sprint Planning:** Agile sprint review covering payment gateway cutovers, load testing, and OpenAPI contracts.
- **🚨 Incident Post-Mortem:** Auth cluster outage review analyzing root cause Redis TLS leaks and preventive action items.
- **🏢 Executive Board:** Strategic review of revenue performance and $2.5M private GPU capital allocation.
- **🤝 Client Discovery:** High-stakes technical scoping with compliance, SCIM identity, and master service agreement dates.

### Live Microphone Recording
1. Click **🎙️ Record Meeting**.
2. Allow browser microphone access when prompted.
3. The real-time Web Audio API visualizer will activate with an ongoing duration timer.
4. Speak normally. When finished, click **Stop Recording**.
5. NeuroMeet automatically passes your speech through Voice Activity Detection, SpeakerNet diarization, Conformer ASR transcription, and neural summarization.

### Uploading Audio Files
- Click **📂 Upload Audio** to select any `.wav` or `.mp3` recording from your local system.
- NeuroMeet processes the entire file and outputs the synchronized diarization timeline and summary.

### Interactive Diarization Timeline
- The horizontal bar at the top represents the meeting duration.
- Each block is color-coded by speaker.
- **Hover** over any block to inspect the speaker's name and turn duration.
- **Click** on any block to automatically scroll the transcript feed directly to that moment!

### AI Meeting Q&A
- Located under the transcript feed.
- Type any question:
  - *"Who is responsible for the Redis patch?"*
  - *"What was decided about the pilot launch?"*
- NeuroMeet searches its dense conversational memory and outputs a factual answer with exact timestamp citations!

---

## 2. Google Meet, Zoom, and MS Teams Chrome Extension

1. Ensure the NeuroMeet API is running (`uvicorn api.main:app --port 8000`).
2. Open your meeting link in Google Chrome.
3. Turn on **Closed Captions** in Google Meet (keyboard shortcut `C`).
4. Notice the floating **NeuroMeet pill** in the bottom-right corner. It tracks speech turns live.
5. Click **Summarize** at any moment to open an in-meeting HUD preview of decisions and action items without leaving your video call!
