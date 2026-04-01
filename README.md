# GPT-2 from Scratch 🤖

A clean, from-scratch PyTorch implementation of GPT-2 (124 M parameters) — built as a learning project to understand the full transformer architecture, from causal self-attention to text generation.

> **Project type:** Personal/educational implementation inspired by Andrej Karpathy's nanoGPT work.

---

## Features

- Full GPT-2 architecture implemented in pure PyTorch:
  - Causal masked multi-head self-attention (Flash Attention via `scaled_dot_product_attention`)
  - GELU-activated feed-forward networks
  - Pre-norm transformer blocks with residual connections
  - Tied token embedding / language-model head weights
- Loads official pretrained weights from HuggingFace (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`)
- Top-k multinomial sampling for text generation
- AdamW optimizer with weight-decay separation and optional fused CUDA variant
- Byte-pair encoding via OpenAI's `tiktoken`

---

## Installation

> **Suggested** — no `requirements.txt` exists yet; install the dependencies below manually.

**Prerequisites:** Python 3.10+, CUDA-capable GPU (recommended, ≥ 8 GB VRAM for the 124 M model)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers tiktoken jupyter
```

Clone the repo:

```bash
git clone https://github.com/Potasiuombun/GPT-2.git
cd GPT-2
```

---

## Usage

### Quickstart

Open the notebook and run all cells in order:

```bash
jupyter notebook gpt2.ipynb
```

| Step | What happens |
|------|-------------|
| Cell 1–2 | CUDA check — verifies your GPU is available |
| Cell 3 | Defines `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, and `GPT` classes |
| Cell 4 | Loads the 124 M pretrained GPT-2 weights from HuggingFace |
| Cell 5 | Moves the model to GPU (`model.eval().to("cuda")`) |
| Cell 6 | Tokenizes a prompt using `tiktoken` and batches it |
| Cell 7 | Runs top-k (k=50) sampling to generate text |

To use a different prompt, edit the string in **Cell 6**:

```python
tokens = enc.encode("Hello, I'm a language model,")
```

### Repo Map

```
GPT-2/
└── gpt2.ipynb   # Full implementation + inference walkthrough
```

---

## Configuration

The model is configured via the `GPTConfig` dataclass (defined in `gpt2.ipynb`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 1024 | Maximum sequence length |
| `vocab_size` | 50257 | BPE vocabulary size |
| `n_layer` | 12 | Number of transformer blocks |
| `n_head` | 12 | Number of attention heads |
| `n_embd` | 768 | Embedding / hidden dimension |

To load a larger pretrained variant, change the argument in Cell 4:

```python
model = GPT.from_pretrained("gpt2-medium")   # 350 M
model = GPT.from_pretrained("gpt2-large")    # 774 M
model = GPT.from_pretrained("gpt2-xl")       # 1558 M
```

> **Note:** Larger models require significantly more GPU memory.

---

## Contributing

This is a personal learning project, but suggestions and improvements are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Commit your changes (`git commit -m "Add my improvement"`)
4. Open a Pull Request

---

## Roadmap

- [ ] Add `requirements.txt` / `environment.yml` for reproducible setup
- [ ] Add a training loop with a small dataset (e.g., Shakespeare)
- [ ] Export model weights to disk and reload without HuggingFace
- [ ] Convert notebook to a standalone Python script (`train.py` / `generate.py`)
- [ ] Add perplexity evaluation on a held-out dataset

---

## FAQ

**Q: Why does text generation run out of memory?**  
A: Running a batch of 5 sequences on a GPU with less than ~8 GB VRAM can hit OOM. Reduce the batch size (`tokens.unsqueeze(0)` for a single sequence instead of `.repeat(5, 1)`) or use a CPU fallback.

**Q: Does this train GPT-2 from scratch?**  
A: The current notebook only loads pretrained weights and runs inference. A training loop is on the roadmap.

**Q: Which GPT-2 variant is used by default?**  
A: `gpt2` (124 M parameters, 12 layers, 12 heads, 768-dim embeddings).

---

## License

No license file is currently present in this repository. All rights are reserved by the author unless stated otherwise. If you would like to reuse this code, please open an issue to request a license.
