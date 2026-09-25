# 🧠 Mastering LLMs

> A structured, prioritized knowledge base covering Large Language Model fundamentals, architecture, data preparation, fine-tuning, evaluation, and deployment.

This repository serves as a comprehensive learning roadmap. It is broken down into four distinct phases, starting from the absolute core mechanics of how text is processed, all the way to advanced optimization and deployment strategies.

---

## 🗺️ The Learning Path

### 🔴 Phase 1: Foundation (10/10)
*The core architecture and mathematical principles that make Large Language Models work.*

* 🧩 **[Tokenization](./Tokenization.md)** — How text is split into tokens and converted into token IDs that an LLM can process.
* 🧮 **[Embeddings](./embeddings.md)** — How tokens and semantic information are represented as continuous vectors for neural networks.
* 🏗️ **[Transformer Architecture](./Transformer%20Architecture.md)** — The foundational components and data flow behind GPT-style language models.
* 🎯 **[QKV Attention](./QKV%20Attention.md)** — The mechanics of Queries, Keys, and Values that allow models to focus on relevant context.
* 🔮 **[Causal LM](./Causal%20LM.md)** — The next-token prediction training objective that drives text generation.

### 🟡 Phase 2: Essential (9/10)
*Data preparation, structuring inputs, and evaluating model intelligence.*

* 💬 **[Prompting and Chat Templates](./Prompting%20Chat%20Templates.md)** — Structuring inputs correctly using model-specific chat formats (e.g., ChatML, Llama 3).
* 🧹 **[Datasets and Data Cleaning](./Datasets%20Data%20Cleaning.md)** — Techniques for collecting, filtering, formatting, and preparing high-quality data.
* 🛠️ **[Fine-Tuning and SFT](./Fine%20Tuning%20SFT.md)** — How Supervised Fine-Tuning (SFT) adapts a base model to follow specific instructions or tasks.
* 📊 **[Evaluation](./Evaluation.md)** — Measuring model performance using benchmarks, metrics, and rigorous error analysis.

### 🟢 Phase 3: Applied (8/10)
*Putting models to work in the real world with external data and efficient serving.*

* 🔍 **[RAG (Retrieval-Augmented Generation)](./RAG.md)** — Grounding LLM responses by dynamically supplying relevant external information.
* 🚀 **[Inference and Serving](./Inference%20Serving.md)** — Strategies to deploy trained models, expose them to applications, and optimize throughput (vLLM, TGI).

### 🔵 Phase 4: Advanced Optimizations
*Techniques for training and running massive models on consumer-grade or limited hardware.*

* ⚡ **[LoRA and PEFT](./LoRA%20PEFT.md)** — Parameter-Efficient Fine-Tuning methods to adapt models without updating the entire network.
* 🗜️ **[QLoRA and Quantization](./QLoRA%20Quantization.md)** — Low-bit precision techniques (4-bit/8-bit) to drastically reduce VRAM requirements during training and inference.

---

## 📂 Repository Index
* 🏠 **[README](./README.md)** — You are here.