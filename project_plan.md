# Project Plan — AI-Powered Meeting Assistant Using Deep Learning

## AAN Mini-Project / Case Study

### Course: Advanced Artificial Intelligence and Neural Networks

---

## 1. Project Title

**"NeuroMeet: A Multimodal Deep Learning Framework for Speaker Diarization, Hierarchical Meeting Summarization, and Action Item Intelligence"**

---

## 2. Project Objectives

1. **Acoustic Signal Processing & Diarization**: Develop an end-to-end speaker diarization pipeline combining Energy/ZCR Voice Activity Detection (VAD), an ECAPA-TDNN neural embedding network (`SpeakerNet`), and Spectral Clustering on the unit hypersphere to accurately attribute meeting turns to individual speakers without pre-enrolled speaker profiles.
2. **Acoustic Transcription**: Implement a Macaron-style Conformer-BiGRU acoustic model (`SpeechCTC`) that maps Log-Mel filterbanks to character sequences via Connectionist Temporal Classification (CTC) decoding.
3. **Hierarchical Attention Summarization**: Overcome information bottlenecking in long, multi-turn dialogues by designing a Hierarchical Attention Network (HAN) combining word-level and utterance-level BiGRU encoders with Scaled Dot-Product and Bahdanau attention.
4. **Pointer-Generator Copy Mechanism**: Implement a dynamic copy gate ($p_{gen}$) to copy named entities, dates, assignees, technical acronyms, and numeric metrics directly from the source transcript, eliminating catastrophic hallucination in critical meeting minutes.
5. **Multi-Task Action Item Extraction**: Formulate action commitment extraction as a joint sequence-tagging and classification task using BiLSTM with Multi-Head Self-Attention to identify tasks, assignees, deadlines, and urgency priorities (`URGENT`, `HIGH`, `MEDIUM`, `LOW`).
6. **Conversational Dynamics & Meeting Health**: Model turn-by-turn sentiment arc, consensus vs contention trajectories, participation balance (Gini inequality index), and a composite Meeting Health Score (0–100) with automated coaching recommendations.
7. **Semantic Memory & Q&A**: Implement a dual-encoder dense retriever (`DenseRetriever`) enabling participants to query historical meeting discussions with cited timestamps and speaker attributions.
8. **Enterprise Delivery**: Provide a production FastAPI asynchronous REST service, an interactive Web Intelligence Studio (`web/index.html`), and a Manifest V3 browser extension for Google Meet, Zoom, and MS Teams.

---

## 3. Experiment Plan & Ablation Matrix

### Experiment Table

| # | Experiment | Hypothesis | Configuration | Primary Metric |
|---|-----------|-----------|---------------|---------------|
| 1 | TextRank Graph Baseline | Lexical sentence centrality provides lower bound without neural generation | `baseline_extractive.yaml` | ROUGE-1 / ROUGE-L |
| 2 | Vanilla Seq2Seq (No Attention) | Fixed-size hidden state causes information bottleneck on long dialogues | Flat BiGRU Encoder-Decoder | ROUGE-1 / BLEU |
| 3 | Hierarchical Attention (HAN) | Decomposing dialogue into words and utterances dramatically improves coherence | `han_summarizer.yaml` | ROUGE-1 / ROUGE-L |
| 4 | Pointer-Generator Copy Net | Dynamic copy gate ($p_{gen}$) eliminates entity corruption on technical terms | `pointer_generator.yaml` | ROUGE-L / Entity Match |
| 5 | Conformer vs BiGRU ASR | Conformer convolution-attention hybrid captures local acoustic formants better | `conformer_ctc.yaml` | WER / CER |
| 6 | ECAPA-TDNN vs GMM Diarization | Multi-scale dilated convolutions with Squeeze-Excitation produce separable d-vectors | `speaker_ecapa_tdnn.yaml` | Diarization Error Rate (DER) |
| 7 | Spectral vs Agglomerative Clustering | Eigengap heuristic accurately determines unknown speaker counts | `diarization.clustering_method` | Speaker Confusion |
| 8 | Multi-Head vs Bahdanau Attention | Multi-Head attention captures orthogonal conversational themes simultaneously | `attention.num_heads` | ROUGE-2 / Attention Entropy |
| 9 | BiLSTM-CRF Action Item Extractor | Joint token-level BIO tagging outperforms rule-based regex on noisy speech | `action_extractor` | Action F1 Score |
| 10 | Dense Retriever vs BM25 Q&A | Dense dual-encoder retrieves semantically paraphrased meeting turns | `dense_retriever` | MRR / Top-1 Accuracy |

---

## 4. Evaluation Strategy

### Metrics Implemented in `src/evaluation/metrics.py`

| Metric | Implementation | Purpose & Target |
|--------|---------------|-----------------|
| **WER** | `word_error_rate(ref, hyp)` | Word-level acoustic transcription accuracy (< 15%) |
| **CER** | `character_error_rate(ref, hyp)` | Character-level edit distance ratio (< 8%) |
| **DER** | `diarization_error_rate(ref, hyp, dur)` | Diarization Error Rate: Missed + False Alarm + Speaker Confusion (< 12%) |
| **ROUGE-1/2/L** | `compute_rouge(ref, hyp)` | Unigram, bigram, and Longest Common Subsequence overlap (> 65.0 ROUGE-1) |
| **BLEU-4** | `compute_bleu(ref, hyp)` | 4-gram precision with brevity penalty (> 40.0 BLEU) |
| **Action F1** | `evaluate_action_items(ref, hyp)` | Harmonic mean of task description and assignee attribution precision/recall (> 90.0%) |

### Standard Test Corpora (`data/processed/`)
- `sample_sprint_planning.json`: Agile technical sprint review, dependency tracking, and cutover dates.
- `sample_tech_postmortem.json`: High-stress production incident post-mortem with root causes and P0 remediations.
- `sample_board_strategy.json`: Financial executive board session reviewing CapEx, revenue growth, and approvals.
- `sample_client_discovery.json`: Enterprise customer discovery call detailing regulatory compliance and onboarding dates.

---

## 5. Error Analysis Framework

1. **Speaker Overlap & Boundary Jitter**: Slicing audio at 1.5s windows with 0.75s stride can cause boundary ambiguity when speakers rapidly interrupt. Mitigated via consecutive turn fusion and VAD silence smoothing.
2. **Out-of-Vocabulary Technical Jargon**: Spoken discussions frequently mention product names ("Stripe v3", "Redis TLS", "B200"). Mitigated by the **Pointer-Generator Copy Gate**, which projects source word locations directly to the output distribution.
3. **Implicit vs Explicit Action Commitments**: Distinguishing between casual musings ("Maybe we could look into that") and hard commitments ("I will deploy the fix by tonight"). Handled by the joint actionability binary sigmoid head gating the token extraction head.
