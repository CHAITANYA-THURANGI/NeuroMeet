# Deep Learning Models & Mathematical Foundations

## 1. Conformer-BiGRU Acoustic Model (`SpeechCTC`)

The acoustic model maps a 2D speech spectrogram $\mathbf{X} \in \mathbb{R}^{B \times 80 \times T}$ into a character sequence $\mathbf{Y}$.

### Macaron-Style Feed-Forward Module
Unlike standard Transformers that place a single feed-forward block after self-attention, the Conformer introduces half-step Feed-Forward modules:
$$\tilde{\mathbf{x}}_i = \mathbf{x}_i + \frac{1}{2} \text{FFN}(\mathbf{x}_i)$$
$$\mathbf{x}'_i = \tilde{\mathbf{x}}_i + \text{MHSA}(\tilde{\mathbf{x}}_i)$$
$$\mathbf{x}''_i = \mathbf{x}'_i + \text{Conv}(\mathbf{x}'_i)$$
$$\mathbf{y}_i = \text{LayerNorm}\left(\mathbf{x}''_i + \frac{1}{2} \text{FFN}(\mathbf{x}''_i)\right)$$

### Connectionist Temporal Classification (CTC) Loss
Given an input sequence $\mathbf{x}$ of length $T$ and target sequence $\mathbf{l}$ of length $U \le T$, CTC defines the conditional probability of $\mathbf{l}$ by marginalizing over all valid alignments $\pi \in \mathcal{B}^{-1}(\mathbf{l})$:
$$P(\mathbf{l} \mid \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} P(\pi \mid \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^T P(\pi_t \mid \mathbf{x}, t)$$
$$\mathcal{L}_{\text{CTC}} = -\ln P(\mathbf{l} \mid \mathbf{x})$$

---

## 2. Speaker Verification & Diarization (`SpeakerNet`)

### Attentive Statistics Pooling (ASP)
Given frame-level representations $\mathbf{h}_t \in \mathbb{R}^C$ for $t \in [1, T]$:
1. Scalar attention weights are computed:
   $$e_t = \mathbf{v}^T \tanh(\mathbf{W} \mathbf{h}_t + \mathbf{b})$$
   $$\alpha_t = \frac{\exp(e_t)}{\sum_{\tau=1}^T \exp(e_\tau)}$$
2. Attention-weighted mean vector:
   $$\boldsymbol{\mu} = \sum_{t=1}^T \alpha_t \mathbf{h}_t$$
3. Attention-weighted standard deviation vector:
   $$\boldsymbol{\sigma} = \sqrt{\sum_{t=1}^T \alpha_t (\mathbf{h}_t - \boldsymbol{\mu}) \odot (\mathbf{h}_t - \boldsymbol{\mu}) + \epsilon}$$
4. Concatenated output representation:
   $$\mathbf{r} = [\boldsymbol{\mu}; \boldsymbol{\sigma}] \in \mathbb{R}^{2C}$$

### Hyperspherical Cosine Embedding
The final linear bottleneck projects $\mathbf{r}$ to a 192-dimensional vector $\mathbf{e}$:
$$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}$$
All speaker verification decisions and spectral affinity graphs operate under standard cosine similarity $\hat{\mathbf{e}}_1^T \hat{\mathbf{e}}_2 \in [-1, 1]$.

---

## 3. Hierarchical Attention Network (`HAN`) with Pointer-Generator

Meetings naturally exhibit a two-level hierarchical structure: words form utterances, and utterances form the meeting narrative.

### Word-Level Encoder
For utterance $i$, word tokens $w_{i,j}$ are embedded and encoded with a bidirectional GRU:
$$\mathbf{h}_{i,j} = \left[\overrightarrow{\text{GRU}}(x_{i,j}); \overleftarrow{\text{GRU}}(x_{i,j})\right]$$
The utterance vector $\mathbf{u}_i$ is pooled using additive Bahdanau attention:
$$\alpha_{i,j} = \text{softmax}_j\left(\mathbf{v}_w^T \tanh(\mathbf{W}_w \mathbf{h}_{i,j} + \mathbf{b}_w)\right)$$
$$\mathbf{u}_i = \sum_{j} \alpha_{i,j} \mathbf{h}_{i,j}$$

### Utterance-Level Encoder
The sequence of utterance vectors $\mathbf{u}_1, \dots, \mathbf{u}_N$ is passed to an utterance-level BiGRU:
$$\mathbf{h}_i^u = \left[\overrightarrow{\text{GRU}}(\mathbf{u}_i); \overleftarrow{\text{GRU}}(\mathbf{u}_i)\right]$$

### Pointer-Generator Copy Gate
At decoder step $t$, the decoder hidden state is $\mathbf{s}_t$ and context vector is $\mathbf{c}_t$.
The probability of generating from the fixed vocabulary versus copying from the input source is parameterized by the generation gate $p_{gen} \in [0, 1]$:
$$p_{gen} = \sigma\left(\mathbf{w}_c^T \mathbf{c}_t + \mathbf{w}_s^T \mathbf{s}_t + \mathbf{w}_x^T \mathbf{x}_t + b_{ptr}\right)$$
Final blended distribution:
$$P(w) = p_{gen} P_{vocab}(w) + (1 - p_{gen}) \sum_{i: w_i = w} a_i^t$$
where $a_i^t$ is the cross-attention weight assigned to turn $i$ at decoder step $t$.
