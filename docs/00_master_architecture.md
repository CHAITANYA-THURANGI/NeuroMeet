# NeuroMeet: Master Architecture Specification

## 1. System Topology & Information Flow

NeuroMeet is designed around a multi-stage, modular pipeline where acoustic signals and spoken discourse are progressively transformed from raw physical vibrations to structured enterprise decisions:

```
+-----------------------------------------------------------------------------------------+
|                                    INPUT MODALITY                                       |
|     [ Raw Audio Waveform (.wav, .mp3) ]    OR    [ Multi-Turn Text Transcript ]         |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|                             ACOUSTIC & SPEECH SUBSYSTEM                                 |
|  1. Voice Activity Detection (Energy & Zero-Crossing Rate Hysteresis State Machine)    |
|  2. Spectral Feature Engineering (80-channel Log-Mel Filterbanks & 13-dim MFCCs)       |
|  3. SpeakerNet (ECAPA-TDNN with Dilated Convolutions & Attentive Statistics Pooling)    |
|  4. Hyperspherical Spectral Clustering (Cosine Affinity + Eigengap Heuristic)           |
|  5. SpeechCTC (Macaron Conformer Blocks + BiGRU Recurrence + CTC Greedy/Beam Decoder)  |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|                              MEETING INTELLIGENCE SUBSYSTEM                             |
|  1. Hierarchical Attention Summarizer (Word-level BiGRU -> Utterance-level BiGRU)       |
|  2. Pointer-Generator Copy Net (p_gen Entity Gate to prevent Named Entity Hallucination)|
|  3. Action Item Classifier (BiLSTM + Multi-Head Self-Attention + BIO Sequence Tagger)   |
|  4. Meeting Dynamics Net (Turn Sentiment, Consensus Scoring, Talk-Time Gini Index)     |
|  5. Dense Semantic Retriever (Dual-Encoder Dense Search for Timelapsed Citations)       |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|                                   DELIVERY PLATFORM                                     |
|  1. Production FastAPI Backend (Async REST Endpoints, OpenAPI /docs)                   |
|  2. Glassmorphic Web Intelligence Studio (Live Audio WebAPI, Waveform, Timeline)        |
|  3. Manifest V3 Chrome Extension (Google Meet, Zoom, MS Teams Closed-Caption Hook)      |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Neural Subsystems

### 2.1 SpeakerNet: Neural Speaker Verification & Diarization
- **Objective:** Convert variable-length acoustic segments into a fixed 192-dimensional vector on the unit sphere such that:
  $$\cos(\mathbf{e}_i, \mathbf{e}_j) \to 1 \quad \text{if same speaker}, \quad \cos(\mathbf{e}_i, \mathbf{e}_j) \to 0 \quad \text{if distinct speakers}$$
- **Time-Delay Neural Network (TDNN):** Utilizes 1D dilated convolutions with receptive field growth across dilation rates $d \in \{1, 2, 3, 4\}$.
- **Squeeze-and-Excitation (SE):** Channel gating dynamically recalibrates spectral filter responses via a bottleneck MLP:
  $$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \text{GAP}(\mathbf{X}))\right)$$
- **Attentive Statistics Pooling (ASP):** Rather than standard temporal averaging, ASP learns a temporal self-attention weight $\alpha_t$ to compute attention-weighted mean $\boldsymbol{\mu}$ and standard deviation $\boldsymbol{\sigma}$.

### 2.2 SpeechCTC: Conformer Acoustic Model
- **Front-end Subsampling:** Strided 1D convolution reduces acoustic frames by a factor of 2, saving quadratic self-attention memory.
- **Macaron Structure:** Each block sandwiches Multi-Head Self-Attention and Depthwise Separable Convolutions between two half-step Feed-Forward Networks.
- **CTC Decoding:** Projects features to character vocabulary and collapses repeating tokens and blanks.

### 2.3 Hierarchical Attention Network (HAN) with Pointer-Generator
- **Word Encoder:** BiGRU transforms token embeddings into word hidden states. Word-level Bahdanau attention pools words into an utterance vector $\mathbf{u}_i$.
- **Utterance Encoder:** BiGRU aggregates the sequence of utterance vectors $\mathbf{u}_1, \ldots, \mathbf{u}_N$ into contextual states $\mathbf{h}_i$.
- **Pointer-Generator Gate ($p_{gen}$):**
  $$p_{gen} = \sigma\left(\mathbf{w}_c^T \mathbf{c}_t + \mathbf{w}_s^T \mathbf{s}_t + \mathbf{w}_x^T \mathbf{x}_t + b_{ptr}\right)$$
  The final output probability distribution blends the vocabulary distribution with the source attention distribution:
  $$P(w) = p_{gen} P_{vocab}(w) + (1 - p_{gen}) \sum_{i: w_i = w} a_i^t$$
  This allows copying technical terms ("Stripe v3", "Redis handshake", "SOC2") directly from speech without vocabulary limitations.
