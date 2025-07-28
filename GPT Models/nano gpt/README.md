# ✍️ nanoGPT: Character-Level Transformer on Shakespeare

This project is a **from-scratch implementation** of Andrej Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT) using **PyTorch tensors and autograd only**—no high-level modules like `nn.Transformer` or `nn.MultiheadAttention`.

It trains a **character-level language model** on Shakespeare’s text and generates eerily coherent English, proving that even small transformer models can capture deep language structure.

---

## 🎯 Goal

To **truly understand** how GPT-style transformers work—not just in theory, but by building every component manually. From how attention flows to how gradients train the model, this project was about mastering the **mechanics of intelligence**.

---

## 🧠 What I Learned

Through this deep dive, I gained **hands-on insight** into each core component of a Transformer:

### 🔑 Self-Attention (Q · Kᵀ → Scores → Softmax → V)
- Built using matrix operations with fused Q, K, V projections.
- Understood how **attention scores** are formed by taking the dot product between Query and Key vectors.
- Softmax on Q·Kᵀ scales focus: some tokens "attend" more based on context.
- These scores weight the Value matrix, extracting the right blend of past token information.

### 🧠 Multi-Head Attention (MHA)
- Instead of looping, all heads were processed in parallel using tensor reshaping—just like modern efficient implementations.
- One large matrix projects input into multiple QKV heads, then we concatenate and project back.
  
### 🔁 Skip Connections + LayerNorm
- Skip connections carry a **residual stream**—the core idea in transformers.
- LayerNorm ensures each layer has stable activations and gradients, promoting smooth learning.

### 🔒 Decoder Masking
- Used masking to prevent a token from attending to future tokens—critical for autoregressive (left-to-right) training.

---

## 🛠️ Manual Implementation Details

- ✅ Used only `torch.tensor`, `autograd`, `F.cross_entropy`, and `softmax`
- ✅ Built `LayerNorm` from scratch
- ✅ Built `MultiHeadAttention` manually (fused QKV, multiple heads in a single matrix)
- ✅ Created our own `Adam` optimizer using Python classes
- ✅ No PyTorch `nn.Module` or pre-built layers—**raw and real**

---

## 📂 File Structure

```bash
.
├── gpt_from_scratch.ipynb     # Full transformer implementation and training
└── README.md                  # You're here

