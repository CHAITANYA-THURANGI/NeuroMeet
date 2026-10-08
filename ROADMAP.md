# NeuroMeet Development Roadmap

## Release Milestones & Strategic Vision

---

### Phase 1: Core Neural Architecture & Engine (v1.0.0 — Current)
- [x] **Acoustic Front-End**: Dual-threshold Energy & Zero-Crossing Rate Voice Activity Detector.
- [x] **Feature Extraction**: Differentiable 80-channel Log-Mel Spectrogram and MFCC computation in PyTorch.
- [x] **Neural Diarization (`SpeakerNet`)**: Time-Delay Neural Network with dilated convolutions, Squeeze-and-Excitation, and Attentive Statistics Pooling (ASP).
- [x] **Hyperspherical Clustering**: Spectral Clustering with automatic eigengap heuristic and cosine affinity matrix.
- [x] **Acoustic CTC Model (`SpeechCTC`)**: Conformer blocks with depthwise separable convolutions, BiGRU recurrent layers, and Connectionist Temporal Classification decoding.
- [x] **Hierarchical Summarization (`HAN`)**: Word-level and utterance-level BiGRU encoders with Scaled Dot-Product and Bahdanau attention mechanisms.
- [x] **Pointer-Generator Copy Net**: $p_{gen}$ copy gate mechanism preserving named entities, metrics, and technical terms.
- [x] **Action Item Extractor**: BiLSTM with Multi-Head Self-Attention for token-level BIO tagging and urgency classification.
- [x] **Dynamics & Health Engine**: Talk-time distribution, Gini inequality coefficient, turn-by-turn sentiment arc, and 0–100 Meeting Health Score.
- [x] **Conversational Memory**: Dual-encoder dense semantic retriever for grounded Q&A with transcript citations.
- [x] **Production REST Service**: FastAPI backend with Pydantic v2 validation, health checks, and OpenAPI documentation.
- [x] **Interactive Web Studio**: Glassmorphic UI with Web Audio API recording, real-time waveform visualizer, speaker diarization timeline, and 1-click Markdown/HTML exports.
- [x] **Manifest V3 Chrome Extension**: Live floating HUD for Google Meet, Zoom, and MS Teams closed-caption capture.
- [x] **Comprehensive Testing Suite**: 43 automated PyTest unit and integration tests passing with 100% success rate.

---

### Phase 2: Streaming & Cloud Integrations (v1.1.0 — Q1 2027)
- [ ] **Bidirectional WebRTC Streaming**: Sub-500ms streaming audio transcription using chunked Conformer inference.
- [ ] **External Model Adapters**: Optional plug-and-play adapter for OpenAI Whisper, Meta MMS, and Google Chirp.
- [ ] **Enterprise Issue Tracker Sync**: Native 1-click synchronization to Atlassian Jira, Linear, GitHub Issues, and Notion Databases.
- [ ] **Calendar & Email Automation**: Automatic calendar invite parsing (Google Calendar / Microsoft Outlook) and automatic meeting minutes email dispatch.

---

### Phase 3: Multilingual & Cross-Lingual Meeting Intelligence (v1.2.0 — Q2 2027)
- [ ] **Multilingual Speech Support**: Expansion to 12 major languages including Hindi, Telugu, Spanish, French, German, and Japanese.
- [ ] **Cross-Lingual Summarization**: Ingest meetings spoken in regional languages and generate structured executive minutes in pristine English.
- [ ] **Code-Mixed Meeting Intelligence**: Seamless handling of code-mixed dialogue (Tanglish, Hinglish, Spanglish) utilizing lessons from the NormMix framework.

---

### Phase 4: Multimodal Vision & Spatial Audio (v2.0.0 — Q4 2027)
- [ ] **Spatial Microphone Array Processing**: Beamforming and direction-of-arrival (DOA) estimation for physical conference room hardware.
- [ ] **Visual Face & Gaze Tracking**: Video stream processing to correlate lip movements with audio for active speaker verification in noisy boardrooms.
- [ ] **Shared Screen Optical Intelligence (OCR)**: Multimodal OCR analyzing slide decks, code snippets, and diagrams presented during screenshares.
