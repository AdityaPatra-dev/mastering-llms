# QLoRA Quantization

 I’ll explain this as a practical chapter rather than just defining terms.

# QLoRA & Quantization — Complete Practical Guide

If you understand this chapter, you should be able to answer:

- What exactly is quantization?
- Why does an LLM use so much VRAM?
- What are FP32, FP16, BF16, INT8 and INT4?
- What does “4-bit model” actually mean?
- What is QLoRA?
- Why does QLoRA save so much VRAM?
- Is QLoRA the same as LoRA?
- Does QLoRA train the quantized model?
- What are NF4, double quantization and paged optimizers?
- How do I choose a model that fits my GPU?
- How do I actually fine-tune a model using QLoRA?
- What happens during training and inference?

---

# 1. The problem QLoRA solves

Suppose you want to fine-tune a **7B parameter LLM**.

A parameter is essentially a learned number.

A 7B model contains approximately:

> **7 billion parameters**

If every parameter is stored using 16 bits:

$$
16\text{ bits}=2\text{ bytes}
$$

Therefore:

$$
7B\times2\ bytes\approx14GB
$$

So just loading the model in FP16/BF16 requires roughly:

**14 GB of memory**

And that's only the model weights.

Training requires additional memory for:

- gradients
- optimizer states
- activations
- temporary tensors
- CUDA overhead

So full fine-tuning can require **far more than 14 GB**.

Your RTX 3060 Ti has:

> **8 GB VRAM**

Therefore, traditional full fine-tuning of a 7B model is impractical on your GPU.

This is where **LoRA** helps.

LoRA says:

> "Don't modify all 7 billion parameters. Keep the original model frozen and train a tiny number of additional parameters."

But there is still a problem.

The original model itself may occupy ~14 GB.

**QLoRA attacks that problem.**

---

# 2. First understand quantization

Quantization means:

> **Representing numbers using fewer bits.**

That's the entire fundamental idea.

Imagine a number:

```text
0.73482194
```

A computer normally represents it using a floating-point format.

Instead of storing it with high precision, we can store an approximation using fewer bits.

For example:

```text
0.735
```

The value isn't exactly the same.

But if the approximation is sufficiently good, the neural network can still work extremely well.

---

# 3. Why do neural networks need so much memory?

Consider:

```text
W = [0.23, -0.81, 0.17, 0.62, ...]
```

These are model weights.

Suppose there are:

```text
7,000,000,000 weights
```

If each weight occupies 16 bits:

```text
7B × 16 bits
```

That's enormous.

But what if we store each weight using only 4 bits?

Then:

```text
7B × 4 bits
```

That's approximately:

$$
7B\times0.5\ bytes
$$

≈ **3.5 GB**

So theoretically:

| Format | Bits/parameter | 7B weights |
|---|---:|---:|
| FP32 | 32 | ~28 GB |
| FP16 | 16 | ~14 GB |
| INT8 | 8 | ~7 GB |
| INT4 | 4 | ~3.5 GB |

These are approximate raw weight sizes.

Real memory usage is somewhat higher because of metadata, buffers, embeddings, temporary tensors, etc.

But the fundamental relationship is:

> **Fewer bits → less memory.**

---

# 4. What exactly is FP32?

FP32 means:

> **32-bit floating point**

A floating-point number has:

```text
sign
exponent
mantissa
```

You don't need to memorize the binary representation yet.

The important thing is:

```text
FP32
↓
32 bits
↓
4 bytes
```

Example:

```text
0.123456789
```

FP32 can represent this with relatively high precision.

---

# 5. FP16

FP16 means:

> **16-bit floating point**

So:

```text
FP32 = 4 bytes
FP16 = 2 bytes
```

This cuts the memory required for weights approximately in half.

Modern GPUs are very good at FP16 computation.

---

# 6. BF16

BF16 = **Brain Floating Point 16**

It is also 16 bits.

Therefore:

```text
BF16 = 2 bytes
```

But its representation differs from FP16.

For deep learning, BF16 is often preferred when hardware supports it because it has a much larger numerical range than FP16.

Conceptually:

```text
FP32
  ↓
high precision

BF16
  ↓
less precision
but large numerical range

FP16
  ↓
less precision
and smaller numerical range
```

For modern training, you'll frequently see:

```python
torch_dtype=torch.bfloat16
```

or:

```python
torch_dtype=torch.float16
```

---

# 7. INT8

Now we move away from floating point.

INT8 means:

> **8-bit integer**

An 8-bit unsigned integer can represent:

$$
0\rightarrow255
$$

A signed INT8 can represent:

$$
-128\rightarrow127
$$

But model weights are floating-point numbers.

So how can we turn:

```text
-0.73
0.21
1.42
```

into integers?

We need **scaling**.

---

# 8. Basic quantization idea

Imagine the original values are:

```text
[-1.0, -0.5, 0.0, 0.5, 1.0]
```

We want to represent them using a small integer range.

Suppose our integer range is:

```text
[-2, -1, 0, 1, 2]
```

We can establish a mapping:

```text
-1.0 → -2
-0.5 → -1
 0.0 →  0
 0.5 →  1
 1.0 →  2
```

Then store:

```text
[-2, -1, 0, 1, 2]
```

instead of:

```text
[-1.0, -0.5, 0.0, 0.5, 1.0]
```

When needed, we approximately reconstruct the original values.

---

# 9. Quantization introduces error

This is extremely important.

Quantization is **lossy**.

Suppose:

```text
original = 0.73
```

After quantization and dequantization:

```text
approximation = 0.75
```

There is an error:

$$
0.75-0.73=0.02
$$

The goal isn't:

> "Make the number exactly identical."

The goal is:

> **Make the error small enough that the model's behavior remains useful.**

---

# 10. Quantization vs compression

They are related but not identical.

Compression generally means:

> Make data occupy less storage.

Quantization specifically means:

> Reduce the numerical precision used to represent values.

For LLMs:

```text
FP16 model
      ↓ quantization
INT8 / INT4 model
```

The model becomes dramatically smaller.

---

# 11. What does "4-bit model" mean?

This phrase causes a lot of confusion.

When someone says:

> "I'm running a 4-bit 7B model."

It generally means the **model weights have been quantized to approximately 4 bits per parameter**.

So instead of:

```text
7B × 16 bits
```

you're approximately doing:

```text
7B × 4 bits
```

That's why 7B models can fit into GPUs that otherwise couldn't hold their FP16 version.

---

# 12. But how can 4 bits represent useful neural-network weights?

This is where sophisticated quantization techniques come in.

You don't simply throw every floating-point number into a crude 16-value bucket.

Modern LLM quantization uses techniques involving things such as:

- scaling
- groups/blocks
- distribution-aware quantization
- specialized data types
- dequantization during computation

And QLoRA specifically uses:

> **NF4**

---

# 13. NF4 — NormalFloat 4

NF4 stands for:

> **4-bit NormalFloat**

This was introduced specifically for quantizing normally distributed neural-network weights.

The important intuition:

LLM weights often have a distribution roughly centered around zero.

Something like:

```text
             █
            ███
          ███████
       █████████████
    ███████████████████
------------------------------
              0
```

Instead of assuming every possible value is equally likely, NF4 uses quantization levels designed around the distribution of weights.

Conceptually:

```text
FP16/BF16 values
       ↓
distribution-aware mapping
       ↓
4-bit representation
```

You don't need to memorize the exact NF4 lookup table to use QLoRA.

Remember:

> **NF4 is a 4-bit datatype designed for efficient neural-network weight quantization.**

---

# 14. Now we can understand LoRA

Before QLoRA, understand LoRA.

Imagine:

```text
Original model

7 billion parameters
```

Instead of training all 7 billion:

```text
freeze original model

       ↓

train small LoRA matrices
```

So:

```text
Original model
     │
     │ frozen
     ↓
LoRA adapter
     │
     │ trainable
     ↓
output
```

This massively reduces trainable parameters.

---

# 15. LoRA mathematically

Suppose a layer contains:

$$
W
$$

Normally training changes:

$$
W\rightarrow W+\Delta W
$$

LoRA says:

> Don't directly learn a gigantic ΔW.

Instead approximate it using two small matrices:

$$
\Delta W=BA
$$

where:

$$
A\in R^{r\times d}
$$

and

$$
B\in R^{d\times r}
$$

where:

$$
r\ll d
$$

So:

```text
Huge update matrix
        ↓
 B × A
        ↓
two tiny matrices
```

This is the core idea of LoRA.

---

# 16. Now combine quantization + LoRA

This gives:

# QLoRA

QLoRA essentially says:

> **Quantize the original model to 4-bit, freeze it, and train LoRA adapters on top of it.**

Conceptually:

```text
                    ┌─────────────────────┐
                    │  Original LLM      │
                    │  7B parameters      │
                    └─────────┬───────────┘
                              │
                         4-bit quantization
                              │
                              ▼
                    ┌─────────────────────┐
                    │  4-bit frozen LLM  │
                    └─────────┬───────────┘
                              │
                       LoRA adapters
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Trainable adapters  │
                    └─────────────────────┘
```

This is the central concept.

---

# 17. The most important misconception

People often say:

> "QLoRA trains the 4-bit model."

Technically, that's not the best way to think about it.

The **quantized base model is frozen**.

The trainable parameters are primarily the:

> **LoRA adapter weights**

So:

```text
Base model
4-bit
Frozen
       +
LoRA
higher precision
Trainable
```

That's the key.

---

# 18. What happens during QLoRA training?

Let's walk through an actual training step.

Suppose your input is:

```text
Explain Docker containers.
```

The input gets tokenized:

```text
Explain
Docker
containers
.
```

The tokens enter the model.

The 4-bit base model performs its computation.

Conceptually:

```text
Input
 ↓
4-bit frozen model
 ↓
LoRA layers
 ↓
prediction
 ↓
loss
```

The loss tells us how wrong the prediction was.

Backpropagation calculates gradients.

But:

```text
Base model gradients → NOT stored/used for updating base weights
```

Instead:

```text
LoRA gradients
      ↓
optimizer
      ↓
LoRA parameters updated
```

Repeat this thousands of times.

---

# 19. But if the model is 4-bit, how does computation happen?

This is another extremely important concept.

The weights may be **stored** in 4-bit format.

But computation doesn't necessarily happen as 4-bit arithmetic everywhere.

Conceptually:

```text
4-bit weights
      ↓
dequantize
      ↓
BF16/FP16 computation
      ↓
result
```

So don't think:

> "The GPU does the entire neural-network computation using tiny 4-bit integers."

That's not generally how QLoRA should be mentally modeled.

Instead:

> **4-bit is primarily a memory-efficient representation of the frozen weights; computations use a suitable higher-precision representation.**

---

# 20. Example

Imagine a quantized weight:

```text
4-bit representation:
1011
```

That is only 4 bits.

During computation, the system can reconstruct an approximate floating-point value:

```text
1011
 ↓
dequantization
 ↓
approximately -0.42
```

Then that value participates in computation.

This is why:

```text
storage precision
```

and

```text
computation precision
```

are different concepts.

---

# 21. Storage vs compute datatype

This distinction will save you a lot of confusion.

Suppose you configure:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

You can interpret it as:

```text
Storage:
    4-bit NF4

Computation:
    BF16
```

So:

```text
4-bit weights
      ↓
dequantization
      ↓
BF16 computation
```

---

# 22. QLoRA's three important ideas

The original QLoRA technique is associated with three major memory-saving ideas:

### 1. 4-bit NF4 quantization

The base model is stored using 4-bit NormalFloat.

### 2. Double quantization

Quantization metadata itself is quantized.

### 3. Paged optimizers

Memory spikes during training are handled more efficiently using paged memory mechanisms.

We'll understand each.

---

# 23. Double quantization

Suppose you quantize:

```text
weights
```

You need additional information to reconstruct them.

For example:

```text
weights
+
scales
+
other quantization metadata
```

Those scales also consume memory.

QLoRA uses:

> **double quantization**

Meaning roughly:

```text
original weights
      ↓
quantize
      ↓
quantization parameters
      ↓
quantize those parameters too
```

So:

```text
weights → quantized
scales  → quantized
```

This saves additional memory.

You generally don't implement this manually.

You enable it with:

```python
bnb_4bit_use_double_quant=True
```

---

# 24. Paged optimizers

Training can have sudden memory spikes.

For example:

```text
normal usage
████████████

temporary spike
████████████████████████
```

If your GPU is already close to its VRAM limit:

```text
CUDA out of memory
```

Paged optimizers help manage optimizer memory using paging mechanisms backed by unified memory.

You may see:

```python
optim="paged_adamw_8bit"
```

This is especially useful when VRAM is limited.

---

# 25. Why QLoRA is so powerful

Compare:

### Full fine-tuning

```text
Base model
      ↓
FP16/BF16
      ↓
all parameters trainable
      ↓
huge memory requirement
```

### LoRA

```text
Base model
      ↓
FP16/BF16
      ↓
frozen
      +
small trainable adapter
```

### QLoRA

```text
Base model
      ↓
4-bit quantized
      ↓
frozen
      +
small LoRA adapter
      ↓
much lower memory requirement
```

That's the progression you should remember.

---

# 26. QLoRA does NOT mean "4-bit LoRA"

That's a useful distinction.

You might think:

```text
QLoRA = LoRA with 4-bit adapter
```

Not quite.

The important architecture is:

```text
4-bit quantized BASE MODEL
+
trainable LoRA adapters
```

The adapters are generally maintained in higher precision suitable for training.

---

# 27. Why don't we quantize the LoRA adapters too?

Because the adapters are the things you're actually learning.

You want stable numerical behavior during optimization.

Conceptually:

```text
Base:
4-bit
frozen

LoRA:
BF16/FP16
trainable
```

This makes sense:

> Store the huge thing cheaply; keep the small trainable thing at useful precision.

---

# 28. Memory breakdown

Suppose you have a 7B model.

Very roughly:

### FP16

```text
7B × 2 bytes
≈ 14 GB
```

### INT8

```text
7B × 1 byte
≈ 7 GB
```

### INT4

```text
7B × 0.5 byte
≈ 3.5 GB
```

But QLoRA training requires more than just the base model.

You also need:

```text
Base model
+
LoRA parameters
+
activations
+
gradients
+
optimizer state
+
temporary memory
+
CUDA overhead
```

Therefore:

> **3.5 GB does NOT mean a 7B QLoRA training job needs only 3.5 GB VRAM.**

This is a very important practical point.

---

# 29. Why sequence length matters

Suppose:

```text
batch size = 1
sequence length = 512
```

Then increase it to:

```text
sequence length = 4096
```

The model has to process much more information simultaneously.

Activations can become a major memory consumer.

Therefore:

```text
longer context
      ↓
more activation memory
      ↓
more VRAM
```

This is why a configuration that works at:

```text
1024 tokens
```

might fail at:

```text
4096 tokens
```

even though the model hasn't changed.

---

# 30. Batch size matters too

Suppose:

```text
batch_size = 1
```

Then increase to:

```text
batch_size = 4
```

You're processing four examples at once.

Generally:

```text
larger batch
    ↓
more activations
    ↓
more VRAM
```

On an 8 GB GPU, you'll frequently need:

```text
per_device_train_batch_size=1
```

and then use:

```text
gradient_accumulation_steps
```

---

# 31. Gradient accumulation

Suppose you want an effective batch size of 8 but can only fit one example.

Use:

```text
batch = 1
gradient accumulation = 8
```

Conceptually:

```text
Example 1 → gradient
Example 2 → gradient
Example 3 → gradient
...
Example 8 → gradient
              ↓
          update model
```

So:

$$
effective\ batch\ size
=
batch\ size\times accumulation\ steps
$$

For example:

$$
1\times8=8
$$

This lets you simulate larger batches without putting all examples into VRAM simultaneously.

---

# 32. QLoRA vs GPTQ vs AWQ vs GGUF

These names can get confusing.

They all relate to quantization, but they're used in different workflows.

### QLoRA

Primarily:

> **quantized model + LoRA fine-tuning**

### GPTQ

A post-training quantization method commonly used for inference.

### AWQ

Another quantization method optimized around preserving important weight information.

### GGUF

A model file format heavily used in the llama.cpp ecosystem.

Often used for local CPU/GPU inference.

The crucial distinction:

```text
QLoRA
→ training/fine-tuning method

GPTQ/AWQ
→ commonly inference quantization methods

GGUF
→ model file format
```

These aren't interchangeable concepts.

---

# 33. QLoRA vs ordinary quantized inference

Suppose you download a 4-bit model and run it.

That's:

> **4-bit inference**

It doesn't automatically mean QLoRA.

QLoRA means:

```text
4-bit base model
+
LoRA
+
training
```

So:

```text
4-bit inference
≠
QLoRA
```

---

# 34. QLoRA vs LoRA

This is perhaps the most important comparison.

| | LoRA | QLoRA |
|---|---|---|
| Base model | FP16/BF16 | 4-bit |
| Base trainable? | No | No |
| LoRA trainable? | Yes | Yes |
| Memory | Low | Even lower |
| Main advantage | Parameter-efficient training | Parameter-efficient + memory-efficient |
| Typical use | GPUs with more VRAM | Limited VRAM |

Conceptually:

```text
LoRA:

FP16 model
████████████████
        +
LoRA
██

QLoRA:

4-bit model
████
 +
LoRA
██
```

---

# 35. What QLoRA allows you to do

This is where it becomes relevant to your setup.

You have:

> RTX 3060 Ti — 8 GB VRAM

That makes memory-efficient fine-tuning particularly interesting.

Instead of trying to fully fine-tune:

```text
7B model
```

you can potentially do:

```text
7B/8B model
 ↓
4-bit quantization
 ↓
LoRA adapters
 ↓
SFT
```

with carefully chosen:

- sequence length
- batch size
- gradient accumulation
- LoRA rank
- gradient checkpointing
- optimizer
- compute dtype

The exact model/context combination determines whether it fits.

---

# 36. Gradient checkpointing

This is another technique you'll encounter constantly.

Normally the model stores many intermediate activations:

```text
Layer 1 → activation
Layer 2 → activation
Layer 3 → activation
...
Layer 32 → activation
```

This consumes VRAM.

Gradient checkpointing says approximately:

> Don't store everything. Recompute some activations when needed during backward propagation.

So:

```text
VRAM ↓
computation ↑
```

It's a classic:

> **trade compute for memory**

With limited VRAM, this can be extremely useful.

---

# 37. A practical QLoRA configuration

A typical Hugging Face setup looks conceptually like this:

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

This means:

```text
load model in 4-bit
        ↓
use NF4
        ↓
double quantization
        ↓
compute using BF16
```

Then:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)
```

---

# 38. Then add LoRA

Using PEFT:

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    task_type="CAUSAL_LM",
)
```

Now you have:

```text
4-bit base model
       +
LoRA adapters
```

---

# 39. What is `r`?

This is LoRA rank.

For example:

```python
r=16
```

Remember:

$$
\Delta W=BA
$$

The rank controls the size/capacity of the low-rank update.

Small:

```text
r = 4
r = 8
```

Medium:

```text
r = 16
r = 32
```

Larger:

```text
r = 64
r = 128
```

Higher rank generally means:

```text
more trainable parameters
+
more adapter capacity
+
more memory/computation
```

but doesn't automatically mean better results.

---

# 40. What is `lora_alpha`?

You'll frequently see:

```python
r=16
lora_alpha=32
```

LoRA applies a scaling factor related to:

$$
\frac{\alpha}{r}
$$

So if:

```text
alpha = 32
r = 16
```

then:

$$
\frac{32}{16}=2
$$

The exact implementation details depend on the LoRA variant/configuration, but the practical idea is:

> `lora_alpha` controls the strength/scaling of the LoRA update.

---

# 41. What are target modules?

This tells LoRA:

> **Which model layers should receive LoRA adapters?**

For a Transformer, you'll often see:

```text
q_proj
k_proj
v_proj
o_proj
```

These correspond to attention projections.

You may also encounter:

```text
gate_proj
up_proj
down_proj
```

which belong to the MLP/feed-forward portion.

For some models, using:

```text
q_proj
k_proj
v_proj
o_proj
```

is a reasonable starting point.

For others, broader target modules may work better.

You must check the architecture of your specific model.

---

# 42. What does LoRA actually modify?

Suppose the original attention projection is:

$$
W_q
$$

LoRA effectively gives:

$$
W'_q=W_q+\Delta W_q
$$

where:

$$
\Delta W_q=BA
$$

During training:

```text
Wq → frozen

A → trainable
B → trainable
```

So you're learning:

```text
"What modification should I make to the existing model?"
```

rather than:

```text
"How do I relearn the entire model?"
```

---

# 43. Why QLoRA can preserve the base model

After training, you generally have:

```text
Base model
+
adapter
```

The adapter can be tiny compared with the base model.

For example:

```text
Base model:
~7 billion parameters

LoRA:
perhaps millions of trainable parameters
```

Therefore you can store:

```text
base model once
+
multiple adapters
```

For example:

```text
base_model
│
├── coding_adapter
├── medical_adapter
├── legal_adapter
├── college_adapter
└── devops_adapter
```

You don't need five complete 7B models.

This is one of the coolest practical properties of LoRA/QLoRA.

---

# 44. Your future architecture

Your earlier idea of separate adapters fits this concept very well.

You could conceptually have:

```text
                 Base LLM
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
   Coding        DevOps       College
   LoRA           LoRA          LoRA
```

Then your RAG system can provide knowledge while the adapter changes behavior/style/task specialization.

For example:

```text
Base model
+
Coding LoRA
+
your coding RAG
```

or:

```text
Base model
+
DevOps LoRA
+
Docker/Kubernetes RAG
```

This is a powerful architecture.

---

# 45. LoRA vs RAG

Don't confuse them.

### RAG

Changes:

> **What information the model has access to at inference time.**

Example:

```text
Your PDF
 ↓
embedding
 ↓
vector database
 ↓
retrieve relevant chunks
 ↓
LLM
```

### LoRA

Changes:

> **The model's learned behavior/parameters.**

Example:

```text
base model
+
coding LoRA
```

So:

```text
RAG = external knowledge

LoRA = learned adaptation
```

They can be combined.

---

# 46. Quantization vs LoRA vs QLoRA

Memorize this table:

| Technology | Main purpose |
|---|---|
| Quantization | Reduce model memory |
| LoRA | Reduce trainable parameters |
| QLoRA | Combine quantized base + LoRA training |
| RAG | Give model external knowledge |
| SFT | Teach model using supervised examples |
| PEFT | Family of parameter-efficient training methods |

This distinction will make your entire LLM learning path much clearer.

---

# 47. QLoRA + SFT

This is probably the workflow you'll use most.

Suppose you have:

```text
Dataset:

instruction
response
```

Example:

```text
User:
How do I create a Docker image?

Assistant:
First create a Dockerfile...
```

You want the model to learn this behavior.

The pipeline becomes:

```text
Dataset
   ↓
Tokenizer
   ↓
QLoRA
   ↓
4-bit frozen base model
   +
LoRA adapters
   ↓
SFT training
   ↓
trained LoRA adapter
```

So:

> **QLoRA is the memory-efficient mechanism; SFT is the supervised learning objective/workflow.**

They aren't competitors.

---

# 48. The complete practical pipeline

Here's the mental model I want you to retain:

```text
                PRETRAINED MODEL
                       │
                       ▼
                4-bit quantization
                       │
                       ▼
              Frozen base model
                       │
                       │
                 Add LoRA
                       │
                       ▼
              Trainable adapters
                       │
                       ▼
                   SFT data
                       │
                       ▼
                Forward pass
                       │
                       ▼
                     Loss
                       │
                       ▼
                 Backpropagation
                       │
                       ▼
              Update LoRA only
                       │
                       ▼
              Save LoRA adapter
```

That is QLoRA fine-tuning.

---

# 49. What actually gets saved?

This is important.

After training you can save:

```text
adapter_model.safetensors
adapter_config.json
```

rather than another complete 7B model.

Then:

```text
Base model
     +
Adapter
     ↓
Fine-tuned behavior
```

You can later merge the adapter with the base model if appropriate.

Or keep them separate.

---

# 50. Adapter merging

Suppose:

```text
W
```

is the base model weight.

LoRA learned:

$$
\Delta W=BA
$$

You can conceptually create:

$$
W'=W+\Delta W
$$

This produces a standalone modified model.

But keeping the adapter separate is often convenient because:

```text
one base model
+
many adapters
```

is storage-efficient.

---

# 51. What happens during inference?

Suppose you've trained:

```text
coding_adapter
```

At inference:

```text
User prompt
     ↓
Base 4-bit model
     +
Coding LoRA
     ↓
Response
```

The adapter modifies the behavior of the base model.

You don't need to perform training again.

---

# 52. A practical Hugging Face stack

For your learning path, understand these libraries:

```text
PyTorch
   ↓
Transformers
   ↓
PEFT
   ↓
bitsandbytes
   ↓
TRL
```

Their roles:

### PyTorch

Underlying deep-learning framework.

### Transformers

Loads/runs Hugging Face models.

### PEFT

Provides LoRA and other parameter-efficient techniques.

### bitsandbytes

Provides common quantization and memory-efficient components.

### TRL

Useful for training workflows such as SFT.

So your QLoRA stack often looks like:

```text
PyTorch
  ↓
Transformers
  ↓
bitsandbytes ← 4-bit
  ↓
PEFT         ← LoRA
  ↓
TRL          ← SFT training
```

---

# 53. A more complete example

A modern workflow can look approximately like:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig
```

Quantization:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

Load:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
```

Tokenizer:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

LoRA:

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    task_type="CAUSAL_LM",
)
```

Then you pass the model + dataset + LoRA configuration into an SFT training workflow.

The exact training API changes across versions, so when you actually build the project, use the current documentation for the installed versions rather than blindly copying an old tutorial.

---

# 54. What about your RTX 3060 Ti?

Your GPU:

```text
RTX 3060 Ti
8 GB VRAM
```

means you should think:

```text
Full fine-tuning
      ↓
generally impractical for 7B+

LoRA
      ↓
possible depending on base model/memory

QLoRA
      ↓
much more practical
```

But **8 GB does not guarantee that every 7B QLoRA setup will fit**.

You have to control:

```text
model size
+
sequence length
+
batch size
+
LoRA configuration
+
gradient checkpointing
+
optimizer
+
compute dtype
```

---

# 55. A realistic 8 GB strategy

If you're experimenting locally, start conservatively.

Something conceptually like:

```text
Model: ~7B/8B
Quantization: 4-bit NF4
Batch: 1
Gradient accumulation: 8–16
Sequence length: 512–1024 initially
LoRA rank: 8–16
Gradient checkpointing: enabled
```

Then measure VRAM.

If it fits comfortably:

```text
sequence length ↑
```

or:

```text
LoRA rank ↑
```

one variable at a time.

Don't immediately jump to:

```text
8192 context
batch 4
rank 64
```

on an 8 GB GPU.

---

# 56. Why context length can destroy your VRAM

Suppose the model itself consumes:

```text
~4 GB
```

You might think:

```text
8 GB - 4 GB = 4 GB available
```

Great.

Then you increase:

```text
sequence length
```

and suddenly:

```text
activations
attention memory
temporary buffers
gradients
```

consume the remaining VRAM.

Result:

```text
CUDA out of memory
```

So when debugging QLoRA OOM errors, don't only look at model size.

---

# 57. The four major memory consumers

For training, think:

```text
VRAM
│
├── Model weights
├── Activations
├── Gradients
└── Optimizer states
```

Plus temporary CUDA allocations.

QLoRA primarily attacks:

```text
MODEL WEIGHTS
```

LoRA attacks:

```text
TRAINABLE PARAMETERS
```

Gradient checkpointing attacks:

```text
ACTIVATION MEMORY
```

Paged optimizers help with:

```text
OPTIMIZER MEMORY SPIKES
```

This is a very useful mental model.

---

# 58. What QLoRA does NOT solve

QLoRA doesn't magically make:

```text
100B model
```

fit comfortably into:

```text
8 GB VRAM
```

Quantization dramatically reduces memory, but there are still:

- activations
- KV cache
- temporary tensors
- computation requirements
- CPU RAM requirements
- GPU memory limits

There is no free lunch.

---

# 59. Quantization during inference

Suppose you have:

```text
8B model
```

FP16:

```text
~16 GB weights
```

4-bit:

```text
~4 GB raw weights
```

Now your GPU can potentially load it.

This is why local LLM tools such as Ollama and llama.cpp can run surprisingly large models on consumer hardware.

Quantization is one of the major reasons local LLMs are practical.

---

# 60. Quantization quality trade-off

Generally:

```text
higher precision
      ↓
more memory
      ↓
potentially better numerical fidelity
```

and:

```text
lower precision
      ↓
less memory
      ↓
greater quantization error
```

But the relationship isn't:

```text
4-bit = terrible
8-bit = good
16-bit = perfect
```

Modern quantization methods can preserve model quality surprisingly well.

The actual impact depends on:

- model architecture
- quantization method
- calibration/data
- task
- context length
- evaluation method

---

# 61. Why NF4 is especially relevant to QLoRA

You may see:

```python
bnb_4bit_quant_type="nf4"
```

and wonder:

> "Why not ordinary INT4?"

Because neural-network weights aren't arbitrary integers.

They tend to have statistical structure.

NF4 uses quantization levels designed around normally distributed weights.

So:

```text
ordinary 4-bit quantization
          vs
distribution-aware 4-bit quantization
```

NF4 is designed to preserve information efficiently in this setting.

---

# 62. Quantization error

Imagine:

```text
Original:

0.01
0.13
0.19
0.24
0.31
0.42
```

After aggressive quantization:

```text
0.00
0.12
0.20
0.25
0.32
0.40
```

Each value has some error.

The network contains billions of weights, so individual errors accumulate.

Good quantization tries to minimize the overall impact.

This is why quantization isn't simply:

```python
round(weight)
```

---

# 63. Calibration

Some quantization approaches use representative data to understand the behavior/distribution of the model.

Conceptually:

```text
model
 ↓
representative data
 ↓
observe activations/weights
 ↓
choose quantization parameters
 ↓
quantized model
```

Different quantization methods have different calibration strategies.

You don't need to deeply understand calibration to start QLoRA, but it becomes important when studying advanced quantization.

---

# 64. Weight quantization vs activation quantization

Another important distinction.

### Weight quantization

Quantize:

```text
model weights
```

### Activation quantization

Quantize:

```text
intermediate activations
```

### KV-cache quantization

Quantize:

```text
attention key/value cache
```

These are different.

QLoRA primarily concerns **4-bit quantization of the frozen base model's weights**.

---

# 65. Why KV cache matters

During generation:

```text
Prompt
 ↓
Transformer
 ↓
attention
```

The model stores key/value information for previous tokens.

That's the:

> **KV cache**

As context grows:

```text
more tokens
 ↓
larger KV cache
 ↓
more memory
```

So you might have a 4-bit model whose weights fit in VRAM but whose **long-context inference** still runs out of memory because the KV cache becomes large.

Again:

> Model weight memory ≠ total inference memory.

---

# 66. Quantization during training vs inference

This distinction is important.

### QLoRA training

```text
4-bit frozen base
+
LoRA
+
higher-precision computation
```

### Quantized inference

```text
quantized model
↓
generate
```

### Full-precision training

```text
FP16/BF16 model
↓
all weights trainable
```

Different workflows.

---

# 67. A complete example in your head

Imagine:

```text
Qwen 7B
```

You want to make it better at your coding style.

### Step 1

Download pretrained model.

```text
Qwen 7B
```

### Step 2

Load using:

```text
4-bit NF4
```

### Step 3

Freeze base model.

```text
7B parameters
↓
not trainable
```

### Step 4

Attach LoRA.

```text
millions of trainable parameters
```

### Step 5

Prepare your dataset.

```text
instruction → response
```

### Step 6

SFT training.

```text
input
 ↓
4-bit base
 ↓
LoRA
 ↓
loss
 ↓
update LoRA
```

### Step 7

Save adapter.

```text
coding_adapter/
```

### Step 8

Later:

```text
Qwen 7B
+
coding_adapter
```

Now you have your specialized model behavior.

---

# 68. What if you want multiple specializations?

This is where your planned architecture becomes interesting.

Imagine:

```text
Qwen 7B base
│
├── coding LoRA
│
├── DevOps LoRA
│
├── Flutter LoRA
│
└── college assistant LoRA
```

Then:

```text
Base + coding
```

for programming.

Or:

```text
Base + DevOps
```

for Docker/Kubernetes.

And use RAG separately for current/reference information.

This avoids creating:

```text
4 × full 7B models
```

---

# 69. How big is a LoRA adapter?

It depends on:

- model architecture
- target modules
- rank
- number of layers

But it can be **orders of magnitude smaller** than the base model.

For example:

```text
Base:
~7B parameters

LoRA:
perhaps tens of millions
```

instead of billions.

This is why adapters are convenient to share.

---

# 70. What if I want to train a 30B model?

This is where your local setup becomes much more constrained.

A 30B model in 4-bit has a rough raw-weight requirement of:

$$
30B\times0.5\ bytes
\approx15GB
$$

That's already substantially beyond 8 GB VRAM.

QLoRA can potentially use:

```text
GPU + CPU offloading
```

but training performance can become much slower, and memory requirements don't disappear.

Therefore:

```text
7B/8B QLoRA
→ realistic starting point

30B QLoRA on 8 GB
→ much more difficult
```

Your 32 GB system RAM can help with offloading, but it doesn't turn 8 GB VRAM into 32 GB of VRAM.

---

# 71. What does CPU offloading mean?

Suppose:

```text
GPU VRAM = 8 GB
RAM = 32 GB
```

You can't fit everything on GPU.

Some components can be placed in system RAM:

```text
CPU RAM
████████████████████

GPU VRAM
████████
```

Data moves between CPU and GPU.

The downside:

```text
CPU ↔ GPU transfer
        ↓
slower
```

So offloading is a memory workaround, not a magical performance improvement.

---

# 72. The most important practical parameters

When you eventually run QLoRA, you'll repeatedly encounter:

```python
load_in_4bit=True
```

```python
bnb_4bit_quant_type="nf4"
```

```python
bnb_4bit_use_double_quant=True
```

```python
bnb_4bit_compute_dtype=torch.bfloat16
```

```python
r=16
```

```python
lora_alpha=32
```

```python
lora_dropout=0.05
```

```python
target_modules=[...]
```

```text
batch size
gradient accumulation
sequence length
gradient checkpointing
optimizer
learning rate
```

These are the knobs you'll actually tune.

---

# 73. A good starting configuration for learning

For a limited-VRAM experiment, conceptually start around:

```text
4-bit:
    NF4

Double quant:
    enabled

Compute:
    BF16 if supported

LoRA:
    r = 8 or 16

Batch:
    1

Gradient accumulation:
    8–16

Sequence:
    512–1024

Gradient checkpointing:
    enabled
```

Then benchmark.

Don't treat these numbers as universal optimal values.

They're starting points.

---

# 74. How to diagnose CUDA OOM

Suppose training crashes:

```text
CUDA out of memory
```

Don't immediately switch models.

Reduce memory systematically:

### First

Reduce:

```text
sequence length
```

### Then

Check:

```text
batch size
```

### Then

Enable:

```text
gradient checkpointing
```

### Then

Reduce:

```text
LoRA rank
```

### Then

Consider:

```text
optimizer / paged optimizer
```

### Then

Consider:

```text
smaller model
```

This gives you a structured debugging process.

---

# 75. What does "4-bit loading" actually save?

Suppose:

```text
7B model
```

FP16:

```text
~14 GB
```

4-bit:

```text
~3.5 GB raw
```

You have saved roughly:

```text
10.5 GB
```

on the raw weights.

That's enormous.

This is the fundamental reason QLoRA changed practical fine-tuning on consumer GPUs.

---

# 76. But why not always use 2-bit?

Because more aggressive quantization introduces more information loss.

Conceptually:

```text
16-bit
████████████████
        ↓
8-bit
████████
        ↓
4-bit
████
        ↓
2-bit
██
```

Memory decreases.

But preserving model behavior becomes increasingly difficult.

There is a trade-off:

$$
\text{memory efficiency}
\leftrightarrow
\text{information preservation}
$$

Modern research continues to push this boundary.

---

# 77. A useful mental analogy

Imagine a high-resolution photograph.

Original:

```text
████████████████████████
```

You compress it.

```text
██████████████
```

Compress harder:

```text
████████
```

At some point:

```text
████
```

The file is tiny, but details disappear.

Quantization is similar conceptually:

```text
FP16
 ↓
8-bit
 ↓
4-bit
 ↓
2-bit
```

The challenge is finding a representation that removes unnecessary numerical precision without destroying useful model behavior.

---

# 78. The terminology you should know

### Quantization

Reducing numerical precision.

### Dequantization

Converting a quantized representation back into a higher-precision representation for computation.

### Calibration

Using model/data information to determine useful quantization parameters.

### NF4

4-bit NormalFloat designed for neural-network weight distributions.

### Double quantization

Quantizing quantization parameters as well.

### QLoRA

4-bit quantized frozen base + trainable LoRA adapters.

### LoRA

Low-rank trainable updates.

### PEFT

Parameter-Efficient Fine-Tuning.

### SFT

Supervised Fine-Tuning.

### Adapter

Small set of trainable parameters added to a pretrained model.

---

# 79. One diagram to remember everything

If you remember only one diagram, remember this:

```text
                         PRETRAINED LLM
                              │
                              │
                       4-bit quantization
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Frozen base model  │
                   │      NF4 / 4-bit    │
                   └──────────┬──────────┘
                              │
                     dequantize for compute
                              │
                              ▼
                         BF16 / FP16
                              │
                              ▼
                     Transformer layers
                              │
                   ┌──────────┴──────────┐
                   │                     │
              Base weights          LoRA adapter
               FROZEN                TRAINABLE
                   │                     │
                   └──────────┬──────────┘
                              ▼
                           Output
                              │
                              ▼
                            Loss
                              │
                              ▼
                      Backpropagation
                              │
                              ▼
                    Update ONLY LoRA
```

That is QLoRA.

---

# 80. QLoRA in one sentence

If someone asks you:

> **"What is QLoRA?"**

You should be able to say:

> **QLoRA is a parameter-efficient fine-tuning technique where a pretrained language model is loaded in low-bit quantized form—typically 4-bit NF4—kept frozen, and trained through small LoRA adapters, while computation is performed in a higher-precision format such as BF16/FP16.**

That's the technically useful definition.

---

# 81. And quantization in one sentence

> **Quantization reduces the number of bits used to represent neural-network values, greatly reducing memory requirements at the cost of some numerical precision.**

---

# 82. How everything you've learned fits together

Your LLM learning sequence is now becoming:

```text
Neural Networks
      ↓
Transformers
      ↓
Tokenization
      ↓
Embeddings
      ↓
Attention / QKV
      ↓
Causal LM
      ↓
Pretraining
      ↓
SFT
      ↓
PEFT
      ↓
LoRA
      ↓
Quantization
      ↓
QLoRA
      ↓
Practical fine-tuning
```

And then:

```text
QLoRA
  +
RAG
  +
Agents/tools
  +
evaluation
```

gets you toward the kind of local AI systems you've been interested in building.

---

# 83. What you should be able to do after this

Before moving on, make sure you can explain these without looking them up:

### Level 1 — fundamentals

- Why does a 7B FP16 model need ~14 GB just for weights?
- Why does 4-bit reduce memory?
- What is quantization error?
- What are FP16/BF16/INT8/INT4?

### Level 2 — QLoRA

- What is NF4?
- Why is the base model frozen?
- Where are the trainable parameters?
- Why does QLoRA require less VRAM than LoRA?
- What is double quantization?
- What are paged optimizers?

### Level 3 — practical

You should understand this configuration:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

and this:

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[...],
)
```

You don't need to memorize the code. You need to know **what every parameter is doing**.

---

# 84. Your next practical step

At this point, **don't spend another 10 hours watching theory videos about QLoRA**.

The best next step is to actually do this:

```text
1. Pick a small 1B–3B model
        ↓
2. Load it in 4-bit
        ↓
3. Inspect VRAM usage
        ↓
4. Attach LoRA
        ↓
5. Prepare a tiny instruction dataset
        ↓
6. Run SFT with QLoRA
        ↓
7. Save the adapter
        ↓
8. Load base + adapter
        ↓
9. Compare before vs after
        ↓
10. Then move to a 7B/8B model
```

That progression will make **quantization + LoRA + SFT + PEFT** click much more deeply than simply reading more theory.

And one particularly important distinction to carry into your next practical project is:

```text
SFT   = WHAT training objective/workflow you're doing
LoRA  = HOW you make only a small part trainable
QLoRA = HOW you make the frozen base model memory-efficient
RAG   = HOW you provide external knowledge
```

Once these four are clear, modern Hugging Face fine-tuning becomes much less confusing.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)