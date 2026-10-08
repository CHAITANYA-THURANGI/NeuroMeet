# Real-Time Live Meeting Deployment & Usage Guide

This guide explains how to use **NeuroMeet AI** in real life during **live meetings** (Google Meet, Zoom, Microsoft Teams) containing **presentations, video, slides, and multi-speaker group discussions**.

---

## 1. How Real-Time Meetings Work (Architecture Overview)

A real corporate meeting contains two simultaneous media streams:
1. **Multi-Speaker Speech & Dialogue:** Group members (Alice, Bob, Charlie) discussing roadmap, technical blockers, deadlines, and decisions.
2. **Visual Presentation & Slides:** Screen shares, slides, documentation, and code reviews.

```
+------------------------------------------------------------------------------------+
|                         LIVE MEETING (Google Meet / Zoom / Teams)                 |
+------------------------------------------------------------------------------------+
         |                                                        |
         v                                                        v
 [Audio Stream (Group Speech)]                       [Visual Screen / Tab / Slides]
         |                                                        |
         +---------------------------+----------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |   NeuroMeet Ingestion Options        |
                  |  1. Chrome Extension HUD (In-Meet)   |
                  |  2. Web Studio Screen / Tab Capture  |
                  |  3. Video / Audio File Upload (MP4)  |
                  +--------------------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |       NeuroMeet Deep Learning Core   |
                  |                                      |
                  |  - VAD (Energy + ZCR Voicing)        |
                  |  - ECAPA-TDNN Speaker Diarization    |
                  |  - Whisper / Conformer ASR           |
                  |  - Hierarchical Attention (HAN)      |
                  |  - BiLSTM-Attention Action Classifier|
                  |  - MeetingDynamicsNet + Health Score |
                  |  - Dense Dual-Encoder Q&A Engine     |
                  +--------------------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |   Multi-Channel Outputs Generated:   |
                  |                                      |
                  |  * Live Floating HUD Inside Call     |
                  |  * Executive TL;DR & Decisions       |
                  |  * Action Items Table (Task/Who/Due) |
                  |  * Talk-Time & Dominance Analytics   |
                  |  * Markdown / HTML / JSON Exports    |
                  |  * 1-Click Copy to Slack / Teams     |
                  |  * Interactive AI Meeting Memory Q&A |
                  +--------------------------------------+
```

---

## 2. Real-Life Usage Method 1: The Chrome Extension (Like NormMix)

Just like your **NormMix** extension was used directly inside the browser, the **NeuroMeet Chrome Extension** attaches directly to Google Meet, Zoom Web, or Microsoft Teams.

### Step 1: Install the Extension in Google Chrome
1. Open Google Chrome and go to `chrome://extensions/`.
2. Enable **"Developer mode"** in the top-right corner.
3. Click **"Load unpacked"** in the top-left corner.
4. Select the directory:
   ```
   C:\projects\NeuroMeet\chrome-extension
   ```
   *(Alternatively, download `neuromeet-chrome-extension-v1.0.0.zip` from the Web Studio header and extract it).*
5. The **NeuroMeet AI** icon will appear in your Chrome toolbar.

### Step 2: Join Your Meeting
1. Open any Google Meet (`https://meet.google.com/`), Zoom (`https://app.zoom.us/`), or Microsoft Teams call.
2. Ensure live captions are turned on in Google Meet (press `c` on your keyboard).
3. Notice the floating **NeuroMeet HUD pill** in the bottom-right corner of your screen showing:
   - `🧠 NeuroMeet [X turns]`
   - Real-time speaker attribution.

### Step 3: Instant Live Insights During the Meeting
- At any point during the call, click **"Summarize"** on the floating HUD.
- The assistant sends the multi-speaker transcript to the local NeuroMeet backend (`http://127.0.0.1:8000`) and opens a modal right over your meeting displaying:
  - **Executive Overview**
  - **Key Decisions**
  - **Action Items with Assignees and Deadlines**
- Click **"Open Full Web Studio"** to jump into the full analytics suite.

---

## 3. Real-Life Usage Method 2: Web Studio Screen & Tab Capture

If you are sharing your screen, viewing a presentation slide deck, or want high-fidelity recording directly from the Web Studio:

1. Open `http://127.0.0.1:8000/`.
2. Click the blue **"🖥️ Share Screen / Meet Tab"** button.
3. Chrome will show the native screen-share dialog:
   - Select the **Chrome Tab** where your Google Meet or Zoom call is running.
   - **Important:** Make sure the **"Also share tab audio"** checkbox is checked.
4. NeuroMeet records both the visual presentation and the audio of all speakers.
5. When the meeting or presentation ends, click **"Stop Screen Capture"**.
6. The entire deep learning pipeline processes the session and displays:
   - Complete speech turns mapped to speakers.
   - Executive minutes.
   - Action item assignments.
   - Meeting health score & participation charts.

---

## 4. Real-Life Usage Method 3: Uploading Recorded Meeting Videos (.mp4, .mkv, .webm)

If your organization records meetings or webinars:

1. Click **"📂 Upload Audio / Video"** in the Web Studio.
2. Select any video or audio format:
   - Videos: `.mp4`, `.mkv`, `.webm`, `.mov`, `.avi`
   - Audio: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`
3. The backend automatically:
   - Extracts the multi-speaker audio track via FFmpeg.
   - Segments speakers via ECAPA-TDNN diarization.
   - Transcribes spoken English via the Whisper acoustic engine.
   - Summarizes the presentation and dialogue via Hierarchical Attention.
   - Categorizes tasks, priorities, and deadlines.

---

## 5. Outputting Results in All Possible Formats

NeuroMeet satisfies every stakeholder consumption requirement:

| Format | How to Use | Best For |
| :--- | :--- | :--- |
| **In-Meeting HUD** | Floating on-screen pill in Chrome | Real-time tracking during the call without leaving the meeting tab |
| **1-Click Clipboard** | Click `📋 Copy to Clipboard` | Immediately pasting formatted minutes and bullet points into **Slack, Microsoft Teams, or Email** |
| **Markdown (`.md`)** | Click `📄 Markdown` | Developer documentation, GitHub Issues, Notion, Obsidian |
| **Printable HTML (`.html`)** | Click `🖨️ HTML Report` | Formal corporate minutes, exporting to PDF (`Ctrl + P`), client delivery |
| **Structured JSON (`.json`)** | Click `📦 JSON Data` | Programmatic integration into Jira, Asana, Linear, CRM, or ERP databases |
| **AI Meeting Memory Q&A** | Type in the **AI Q&A Assistant** drawer | Interactively asking questions like *"Who is assigned to test the payment gateway?"* or *"What was decided about the rollout date?"* |

---

## 6. Real-World Production & Cloud Deployment

To deploy NeuroMeet as a 24/7 service for your entire team or organization:

### A. Docker Container Deployment
Create a `Dockerfile` and run on any cloud VM (AWS EC2, GCP Compute Engine, Azure, RunPod, or DigitalOcean):
```bash
# Build production image
docker build -t neuromeet-ai:latest .

# Run on GPU or CPU instance
docker run -d -p 8000:8000 --gpus all --name neuromeet neuromeet-ai:latest
```

### B. Remote Backend for Chrome Extension
In [chrome-extension/content/content.js](file:///c:/projects/NeuroMeet/chrome-extension/content/content.js#L90-L100) and [chrome-extension/manifest.json](file:///c:/projects/NeuroMeet/chrome-extension/manifest.json#L10-L16), replace `http://127.0.0.1:8000` with your deployed HTTPS endpoint:
```javascript
const API_BASE = "https://neuromeet.yourcompany.com";
```
Now every team member who installs the Chrome extension will automatically connect to your central cloud GPU server!
