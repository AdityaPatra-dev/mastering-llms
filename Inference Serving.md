# Inference Serving

The goal of this explanation is that after reading it, you should understand not only *what inference/serving means*, but also how to actually take a Hugging Face model, run it locally, expose it as an API, optimize it, and eventually build an LLM application around it.

# Inference / Serving — Complete Practical Guide

## 1. The simplest definition

There are two related concepts:

**Inference** = using a trained model to produce an output.

**Serving** = making that model available so that applications/users can send requests to it.

For an LLM:

```text
User:
"Explain Docker"

        ↓

Your application
        ↓
LLM inference server
        ↓
Tokenizer
        ↓
LLM
        ↓
Generated tokens
        ↓
Decoded text

" Docker is a platform..."
```

So:

> **Inference is the computation. Serving is the infrastructure/API around that computation.**

---

# 2. Training vs inference

This distinction is extremely important.

Suppose you have a model:

```text
Qwen / Llama / Mistral
```

During training:

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Compare prediction with correct answer
  ↓
Loss
  ↓
Backpropagation
  ↓
Update weights
```

The model's parameters change.

During inference:

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Output
```

The model's parameters **do not change**.

For example:

```text
Training:

"What is 2 + 2?" → "4"

Model makes mistake
↓
Calculate loss
↓
Update weights
```

Inference:

```text
"What is 2 + 2?"
        ↓
Model
        ↓
"4"
```

No learning occurs.

---

# 3. What exactly happens during LLM inference?

Suppose you send:

```text
Explain Docker in simple terms.
```

The computer doesn't directly give this string to the neural network.

There is a pipeline.

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embeddings
 ↓
Transformer
 ↓
Logits
 ↓
Probability distribution
 ↓
Choose next token
 ↓
Repeat
 ↓
Token IDs
 ↓
Tokenizer decoder
 ↓
Text
```

Let's go through this carefully.

---

# 4. Step 1 — Tokenization

Your input:

```text
Explain Docker in simple terms.
```

gets converted into tokens.

Something conceptually like:

```text
["Explain", " Docker", " in", " simple", " terms", "."]
```

Then those tokens become IDs:

```text
[12345, 8921, 304, 6789, 245]
```

The exact IDs depend on the tokenizer.

So:

```text
Text
 ↓
Tokenizer
 ↓
[12345, 8921, 304, 6789, 245]
```

This is why tokenization is fundamental to inference.

---

# 5. Step 2 — The model receives token IDs

The neural network doesn't understand:

```text
12345
8921
304
```

as meanings.

These IDs are converted into vectors through the model's embedding layer.

Conceptually:

```text
12345 → [0.12, -0.43, 0.81, ...]
8921  → [0.72,  0.11, -0.32, ...]
```

These vectors then go through the Transformer.

---

# 6. Step 3 — Transformer computation

Your model may contain dozens of Transformer layers.

For example:

```text
Input embeddings
       ↓
Transformer layer 1
       ↓
Transformer layer 2
       ↓
Transformer layer 3
       ↓
...
       ↓
Transformer layer 32
       ↓
Output
```

Each layer performs operations involving things like:

- self-attention
- Q/K/V
- matrix multiplication
- MLP/feed-forward layers
- normalization
- residual connections

You've already been learning these pieces.

Inference is basically:

> **Running these learned computations forward without changing the weights.**

---

# 7. Step 4 — Logits

Eventually the model produces something called **logits**.

Imagine the vocabulary contains:

```text
100,000 tokens
```

The model produces approximately:

```text
token       logit

"Docker"      2.1
"is"          5.7
"runs"        3.8
"banana"     -2.4
"container"   7.2
...
```

These aren't probabilities yet.

They are raw scores.

---

# 8. Step 5 — Softmax

The logits can be converted into probabilities.

Conceptually:

```text
container → 0.42
is        → 0.21
Docker    → 0.14
runs      → 0.08
...
```

Now the model has a probability distribution for the next token.

---

# 9. Step 6 — Choose the next token

The model has to decide which token comes next.

Suppose:

```text
"The Docker"

Possible next tokens:

"container" → 0.50
"image"     → 0.25
"command"   → 0.10
"is"        → 0.05
...
```

There are several ways to choose.

This is where **generation parameters** become important.

---

# 10. Greedy decoding

The simplest method:

> Choose the highest-probability token.

Example:

```text
container → 0.50
image     → 0.25
command   → 0.10
```

Choose:

```text
container
```

Then:

```text
"The Docker container"
```

Run the model again to determine the next token.

---

# 11. Autoregressive generation

This is one of the most important concepts in LLM inference.

A causal language model generates:

```text
token 1
 ↓
token 2
 ↓
token 3
 ↓
token 4
 ↓
...
```

For example:

```text
Input:
"Python is"

Model predicts:
" a"

Now:
"Python is a"

Model predicts:
" programming"

Now:
"Python is a programming"

Model predicts:
" language"
```

So inference is iterative.

```text
Prompt
 ↓
predict token
 ↓
append token
 ↓
predict next token
 ↓
append token
 ↓
repeat
```

This is why it's called **autoregressive generation**.

The model uses its own previously generated tokens as context.

---

# 12. Why LLM inference can be expensive

Suppose you generate:

```text
1 token
```

The model has to perform substantial computation.

Now generate:

```text
1,000 tokens
```

You repeatedly run generation.

That's why:

```text
Longer output
       ↓
More computation
       ↓
More latency
```

But modern LLM inference uses an extremely important optimization:

# KV Cache

---

# 13. KV cache

This is one of the most important concepts for practical LLM serving.

Remember attention:

```text
Q = Query
K = Key
V = Value
```

During generation, previously processed tokens don't need to have their K/V representations recomputed from scratch every time.

Instead, the model stores them.

```text
Previous tokens
      ↓
K/V
      ↓
CACHE
```

Then when generating the next token:

```text
New token
   ↓
Q
   ↓
Compare with cached K/V
   ↓
Attention
```

Conceptually:

```text
Without KV cache:

token 1 → calculate
token 1 + token 2 → calculate everything again
token 1 + token 2 + token 3 → calculate everything again
...
```

With KV cache:

```text
token 1 → calculate + cache

token 2 → use cache + calculate token 2

token 3 → use previous cache + calculate token 3

...
```

This dramatically improves autoregressive generation efficiency.

---

# 14. Prefill vs decode

When serving LLMs, you'll often hear:

**Prefill**

and

**Decode**

These are extremely important.

Suppose your prompt is:

```text
Explain Kubernetes in 2,000 words...
```

### Prefill

The model processes the existing prompt.

```text
2000 input tokens
       ↓
Transformer
       ↓
KV cache
```

This is called the **prefill phase**.

Then the model starts generating.

### Decode

```text
Generate token 1
Generate token 2
Generate token 3
...
```

This is the **decode phase**.

So:

```text
Request
   ↓
PREFILL
   ↓
KV cache
   ↓
DECODE
   ↓
token
token
token
token
```

This distinction becomes extremely important when optimizing inference servers.

---

# 15. What does "inference speed" mean?

You will encounter several metrics.

## Tokens per second

For example:

```text
35 tokens/s
```

means the model generates approximately 35 tokens every second.

For an interactive chatbot, this is highly relevant.

---

## Time to First Token — TTFT

Suppose you send:

```text
Hello
```

and wait:

```text
0.8 seconds
```

before seeing the first generated token.

That's approximately:

**TTFT = 0.8 seconds**

It measures responsiveness.

---

## Inter-token latency

After the first token appears:

```text
token 1
 ↓ 30 ms
token 2
 ↓ 30 ms
token 3
 ↓ 30 ms
...
```

This affects how smooth streaming feels.

---

# 16. Latency vs throughput

These are different.

### Latency

How long one request takes.

```text
Request
 ↓
2 seconds
 ↓
Response
```

### Throughput

How many requests/tokens the server can process over time.

Example:

```text
100 requests/minute
```

A server optimized for one user may not be optimized for 100 users.

That's why **serving** is more complicated than simply running a model.

---

# 17. What is model serving?

Imagine you've written:

```python
model.generate(...)
```

That's inference.

But now suppose you want your Flutter application to use it.

You could create:

```text
POST /generate
```

Your application sends:

```json
{
  "prompt": "Explain Docker"
}
```

Your server runs inference:

```text
Prompt
 ↓
Model
 ↓
Generated text
```

and returns:

```json
{
  "response": "Docker is..."
}
```

That's model serving.

---

# 18. Basic architecture

A simple architecture looks like:

```text
                    INTERNET / LAN
                         │
                         ↓
                   ┌────────────┐
                   │   Client   │
                   │ Flutter/Web│
                   └─────┬──────┘
                         │ HTTP
                         ↓
                   ┌────────────┐
                   │ API Server │
                   └─────┬──────┘
                         │
                         ↓
                 ┌───────────────┐
                 │ Inference     │
                 │ Engine        │
                 └───────┬───────┘
                         │
                         ↓
                    ┌─────────┐
                    │   GPU   │
                    │  Model  │
                    └─────────┘
```

---

# 19. Hugging Face inference

At the simplest level, you can load a model using Transformers.

Conceptually:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "..."

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer(
    "Explain Docker",
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=200
)

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(answer)
```

The important pipeline is:

```text
from_pretrained()
      ↓
Tokenizer + Model
      ↓
tokenize()
      ↓
generate()
      ↓
decode()
```

---

# 20. `model.eval()`

When doing inference with PyTorch models, you normally put the model in evaluation mode:

```python
model.eval()
```

This tells relevant layers to behave in inference/evaluation mode.

For example, some training-specific behavior such as dropout changes.

---

# 21. `torch.no_grad()` / inference mode

You also don't want PyTorch calculating gradients during ordinary inference.

Conceptually:

```python
with torch.no_grad():
    output = model(...)
```

Modern PyTorch also provides:

```python
with torch.inference_mode():
    output = model(...)
```

This reduces unnecessary training-related overhead.

The important idea:

```text
Training:
gradient calculation → YES

Inference:
gradient calculation → NO
```

---

# 22. Device placement

Suppose you have:

```text
CPU
GPU
```

You generally want the model on the appropriate accelerator.

For example:

```python
model.to("cuda")
```

and inputs need to be on the appropriate device as well.

Conceptually:

```text
CPU
 ↓
Input preparation
 ↓
GPU
 ↓
Model inference
 ↓
GPU output
 ↓
CPU
 ↓
Text
```

Moving data between CPU and GPU has a cost.

Therefore, good inference systems try to minimize unnecessary transfers.

---

# 23. Why GPU VRAM matters so much

Suppose your model has:

```text
7 billion parameters
```

If represented using FP16:

```text
~2 bytes / parameter
```

Then the raw weights require roughly:

```text
7B × 2 bytes
≈ 14 GB
```

That's before considering other memory requirements.

So a 7B FP16 model generally doesn't fit comfortably into an 8 GB GPU.

This is where quantization becomes extremely useful.

---

# 24. Quantization for inference

Instead of representing weights with:

```text
FP16
```

you can use:

```text
INT8
INT4
```

etc.

Roughly:

```text
FP16
2 bytes / parameter

INT8
1 byte / parameter

INT4
0.5 byte / parameter
```

So a 7B model at 4-bit weights may need roughly:

```text
7B × 0.5
≈ 3.5 GB
```

plus overhead and runtime memory.

That's why quantized models can run on smaller GPUs.

---

# 25. But model weights aren't the entire memory usage

This is extremely important.

GPU memory includes things like:

```text
Model weights
+
KV cache
+
activations / temporary buffers
+
CUDA/runtime overhead
+
batch-related memory
```

So:

```text
Model size ≠ total VRAM requirement
```

This becomes especially important for long-context models.

---

# 26. Context length

Suppose your model supports:

```text
8K context
```

and you send:

```text
8,000 tokens
```

The model needs memory for processing that context.

If you use:

```text
32K
128K
```

context windows, KV-cache memory can become a major part of inference memory usage.

Therefore:

> **Long context can consume substantial memory even when the model weights themselves fit.**

---

# 27. Batch size

Suppose one user sends a request:

```text
batch = 1
```

Now imagine 32 users send requests simultaneously:

```text
batch = 32
```

The server can process requests together.

This can improve hardware utilization and throughput.

But it also increases memory requirements.

So:

```text
larger batch
    ↓
potentially higher throughput
    ↓
higher memory usage
```

There is a tradeoff.

---

# 28. Continuous batching

Traditional serving might do something like:

```text
Request A
 ↓
finish
 ↓
Request B
 ↓
finish
 ↓
Request C
```

This can leave the GPU underutilized.

Modern LLM serving systems can use **continuous batching**.

Imagine:

```text
A ────────────────→
      B ───────────────→
          C ───────────────→
             D ─────────────→
```

The server dynamically combines active requests during generation.

This is one reason specialized inference engines can outperform a naive Python implementation.

---

# 29. Important inference/serving frameworks

You should know the ecosystem.

### Transformers

The basic Hugging Face library.

Good for:

- learning
- experimentation
- custom inference
- research
- building prototypes

---

### Text Generation Inference (TGI)

Hugging Face's inference server technology.

Designed specifically for serving generative models.

Useful concepts include:

```text
HTTP API
streaming
batching
GPU inference
production serving
```

---

### vLLM

A very important serving engine.

It is widely used for LLM inference and focuses heavily on efficient serving.

One major idea associated with vLLM is:

**PagedAttention**

which efficiently manages KV-cache memory.

You don't need to implement PagedAttention yourself initially.

Understand the concept:

```text
Many requests
     ↓
Many KV caches
     ↓
Efficient memory management
     ↓
Higher serving efficiency
```

---

### llama.cpp

Extremely important for local LLMs.

It focuses on efficient inference, particularly for quantized models and CPU/GPU hybrid execution.

You'll encounter formats such as:

```text
GGUF
```

This is especially relevant to your local-LLM goals.

---

### Ollama

Ollama provides a convenient experience for running models locally.

Instead of building the whole serving infrastructure yourself, you can do something conceptually like:

```text
Ollama
 ↓
Model
 ↓
Local API
 ↓
Your application
```

It's excellent for quickly getting a local model running.

---

# 30. Transformers vs vLLM vs Ollama vs llama.cpp

Think about them like this:

| Tool | Main purpose |
|---|---|
| Transformers | Model development/inference |
| vLLM | High-performance LLM serving |
| TGI | Production-oriented HF model serving |
| llama.cpp | Efficient local inference |
| Ollama | Easy local model management + serving |

They overlap, but they target different workflows.

---

# 31. What is an inference engine?

An inference engine is the software responsible for efficiently executing your model.

Instead of:

```text
Python
 ↓
naive model execution
```

you can have:

```text
API
 ↓
Inference engine
 ↓
optimized kernels
 ↓
GPU
```

The engine may optimize:

- GPU kernels
- memory management
- KV cache
- batching
- scheduling
- quantization
- parallelism
- communication

---

# 32. What is streaming?

Without streaming:

```text
User sends request

[wait 5 seconds]

Full response arrives
```

With streaming:

```text
User sends request

0.8 sec → "Docker"
1.0 sec → " is"
1.1 sec → " a"
1.2 sec → " platform"
...
```

The user sees the response being generated.

This makes chat applications feel much more responsive.

Technically, your server might stream tokens through:

```text
HTTP streaming
SSE
WebSocket
```

SSE = **Server-Sent Events**.

For a typical LLM chatbot, SSE is often enough.

---

# 33. API serving

You can build a simple API with something like FastAPI.

Conceptually:

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/generate")
def generate(prompt: str):
    answer = model_generate(prompt)
    return {"response": answer}
```

Then your Flutter application could call:

```text
POST /generate
```

with:

```json
{
    "prompt": "Explain Kubernetes"
}
```

and receive:

```json
{
    "response": "Kubernetes is..."
}
```

That's your first real model-serving architecture.

---

# 34. OpenAI-compatible APIs

An extremely useful concept.

Many inference servers expose APIs that resemble OpenAI's API format.

Then your application can effectively do:

```text
Your application
      ↓
OpenAI-compatible API
      ↓
Local model
```

This is powerful because your application doesn't necessarily need to know whether the backend is:

```text
OpenAI
local vLLM
Ollama
another inference server
```

You can change the backend while keeping much of your application interface the same.

---

# 35. Example architecture for your local LLM setup

For your own local AI system, you could eventually build:

```text
                 ┌─────────────────┐
                 │ Flutter / VSCode│
                 │      Client     │
                 └────────┬────────┘
                          │
                          ↓
                 ┌─────────────────┐
                 │   API Gateway   │
                 └────────┬────────┘
                          │
              ┌───────────┴───────────┐
              ↓                       ↓
       ┌─────────────┐         ┌─────────────┐
       │ Local LLM   │         │ RAG System  │
       │   Server    │         │ Vector DB   │
       └──────┬──────┘         └─────────────┘
              │
              ↓
             GPU
```

And the RAG system could provide context:

```text
Question
 ↓
Retriever
 ↓
Relevant documents
 ↓
Prompt construction
 ↓
LLM server
 ↓
Answer
```

So serving is one of the central pieces connecting your model to the rest of your AI application.

---

# 36. Inference parameters

When using an LLM, you'll encounter parameters such as:

```text
temperature
top_p
top_k
max_new_tokens
stop
repetition_penalty
```

You should understand what they do.

---

# 37. `max_new_tokens`

Controls how many new tokens the model can generate.

Example:

```python
max_new_tokens=100
```

means:

> Generate at most approximately 100 new tokens.

It does **not** mean the entire context is limited to 100 tokens.

For example:

```text
Input = 2000 tokens
max_new_tokens = 500
```

The total sequence can be approximately:

```text
2500 tokens
```

subject to the model's context limit.

---

# 38. Temperature

Temperature controls how much randomness is introduced into sampling.

Conceptually:

### Low temperature

More deterministic:

```text
0.0
0.2
0.3
```

### Higher temperature

More variation:

```text
0.7
1.0
1.5
```

But don't think:

> "Higher temperature = smarter."

It doesn't.

It changes the sampling distribution.

For deterministic technical tasks, lower values are often useful.

For creative generation, higher values may produce more variation.

---

# 39. Top-k

Suppose the model produces:

```text
100,000 possible tokens
```

Top-k might say:

```text
Only consider the top 50 candidates.
```

Then sample from those.

```text
100,000
    ↓
top 50
    ↓
sample
```

---

# 40. Top-p

Top-p is nucleus sampling.

Instead of saying:

```text
choose top 50
```

it says approximately:

> Keep the smallest set of tokens whose cumulative probability reaches p.

For example:

```text
p = 0.9
```

means retain the candidate tokens making up roughly 90% of the probability mass.

---

# 41. Temperature + top-p

These can work together.

Conceptually:

```text
Logits
 ↓
Temperature adjustment
 ↓
Probability distribution
 ↓
Top-p/top-k filtering
 ↓
Sampling
 ↓
Next token
```

You don't need to memorize the exact mathematical implementation immediately.

Understand the pipeline.

---

# 42. Greedy vs sampling

### Greedy

```text
Choose highest probability.
```

Example:

```text
A = 0.60
B = 0.25
C = 0.15

→ A
```

### Sampling

Could choose:

```text
A
```

most of the time, but occasionally:

```text
B
```

depending on the distribution.

This introduces variability.

---

# 43. Deterministic inference

For some applications you want:

```text
same input
    ↓
same output
```

This is useful for:

- classification
- extraction
- structured outputs
- testing
- evaluation

For creative generation you may intentionally allow randomness.

---

# 44. Stop tokens

Suppose your model is generating:

```text
Question:
...

Answer:
...
```

You might tell the generation system to stop when it encounters a particular token/string.

This prevents:

```text
Answer
↓
continues generating unrelated content
```

---

# 45. Structured generation

This is extremely useful in applications.

Instead of asking:

```text
Give me information about this person.
```

and getting free-form text, you can request a structure such as:

```json
{
  "name": "...",
  "age": 20,
  "skills": ["Python", "Docker"]
}
```

Then your application can directly process it.

This is especially useful for:

```text
LLM
 ↓
JSON
 ↓
Python
 ↓
Database
```

---

# 46. Inference with your fine-tuned LoRA model

This connects directly to what you've been learning.

Suppose you have:

```text
Base model
+
LoRA adapter
```

During inference you can load:

```text
Base model
       +
LoRA adapter
       ↓
Fine-tuned behavior
```

Conceptually:

```text
Qwen/Llama
    +
your coding LoRA
    ↓
coding-specialized model
```

You don't necessarily need to train the entire model again.

---

# 47. Merging LoRA

You may also merge the adapter into the base model.

Conceptually:

```text
Base model
    +
LoRA
    ↓
Merged model
```

Or keep them separate:

```text
Base model
 ├── coding LoRA
 ├── medical LoRA
 ├── writing LoRA
 └── another LoRA
```

This makes adapters useful for switching behaviors.

---

# 48. Quantized inference

Suppose your model is:

```text
FP16
```

You can quantize it:

```text
FP16
 ↓
INT8 / INT4
```

Then run inference with reduced memory requirements.

But quantization can involve tradeoffs:

```text
less memory
+
potentially faster inference
-
possible quality degradation
```

The exact effect depends heavily on the model, quantization method, hardware, and workload.

---

# 49. CPU vs GPU inference

### CPU

Advantages:

```text
large system RAM
cheap
easy to run
```

Disadvantages:

```text
generally lower generation speed
```

### GPU

Advantages:

```text
massive parallel computation
high throughput
fast matrix operations
```

Disadvantages:

```text
limited VRAM
expensive
```

---

# 50. Hybrid CPU + GPU inference

This is particularly relevant to your hardware.

Imagine:

```text
Model
 ├── some layers → GPU
 └── some layers → CPU
```

This is called offloading.

For example:

```text
CPU RAM
  ↓
some model layers

GPU VRAM
  ↓
other model layers
```

It allows models larger than your GPU VRAM to run, although performance depends on how much data must move between CPU and GPU.

---

# 51. Your RTX 3060 Ti example

You have approximately:

```text
RTX 3060 Ti
8 GB VRAM
```

That means you need to think carefully about:

```text
model size
quantization
context length
KV cache
GPU layers
CPU offload
```

For example, instead of thinking:

> "Can a 14B model fit in 8 GB?"

think:

```text
14B model
   ↓
What quantization?
   ↓
Weight memory?
   ↓
KV cache?
   ↓
Context length?
   ↓
Runtime overhead?
   ↓
How much GPU offloading?
   ↓
Expected tokens/sec?
```

That's the proper way to reason about inference.

---

# 52. Why 4-bit models are popular locally

Suppose:

```text
14B FP16
```

requires roughly:

```text
28 GB
```

just for weights.

A 4-bit representation might reduce the raw weight requirement toward:

```text
~7 GB
```

before runtime overhead.

That makes a 14B model much more feasible on an 8 GB GPU with careful configuration.

But:

> **7 GB theoretical weight storage does not mean the model will necessarily run comfortably in 8 GB VRAM.**

KV cache and other memory matter.

---

# 53. Model format matters

You may encounter:

```text
safetensors
GGUF
GPTQ
AWQ
bitsandbytes 4-bit
```

Don't confuse these.

Some are primarily:

```text
model file/storage formats
```

while others involve:

```text
quantization schemes/runtime methods
```

For example:

**GGUF** is especially common in the llama.cpp ecosystem.

Hugging Face Transformers often works directly with model repositories containing formats such as SafeTensors and can use various quantization backends.

---

# 54. What does "serving in production" add?

Running:

```bash
python model.py
```

is not production serving.

A production system has to consider:

```text
Authentication
Rate limiting
Logging
Monitoring
Timeouts
Retries
Concurrency
Batching
GPU utilization
Memory management
Model loading
Health checks
Autoscaling
Security
```

Architecture:

```text
Client
 ↓
Load balancer
 ↓
API server
 ↓
Inference scheduler
 ↓
GPU workers
 ↓
Model
```

---

# 55. Health checks

Your server might expose:

```text
GET /health
```

which returns:

```json
{
  "status": "ok"
}
```

This allows infrastructure to determine whether the service is alive.

---

# 56. Logging

You may want to record:

```text
request ID
timestamp
latency
input token count
output token count
model
error
GPU utilization
```

Be careful with logging sensitive user prompts.

Don't blindly store everything.

---

# 57. Authentication

If you expose your model over the internet:

```text
Internet
 ↓
API
 ↓
GPU
```

you should not leave an unrestricted expensive GPU endpoint publicly accessible.

You might use:

```text
API key
JWT
OAuth
network restrictions
reverse proxy
```

---

# 58. Rate limiting

Without rate limiting:

```text
Attacker/user
 ↓
10,000 requests
 ↓
Your GPU
 💀
```

Rate limiting can enforce:

```text
100 requests/minute
```

or token-based limits.

---

# 59. Dockerizing inference

This is directly relevant to your Docker learning.

You can package:

```text
Inference server
+
Python dependencies
+
model runtime
```

into a container.

Conceptually:

```text
Docker container
 ├── API server
 ├── inference engine
 ├── Python libraries
 └── configuration

        ↓

      GPU
```

With NVIDIA GPU support, the container can access the host GPU through the appropriate NVIDIA container runtime/tooling.

---

# 60. Kubernetes serving

Eventually your architecture could become:

```text
Kubernetes
   │
   ├── API deployment
   │
   ├── inference deployment
   │
   ├── GPU nodes
   │
   └── monitoring
```

For a student project, you absolutely don't need Kubernetes initially.

But this is where:

```text
LLM
+
Docker
+
Kubernetes
+
cloud
```

connect.

---

# 61. Multi-model serving

Suppose you have:

```text
Qwen 14B
Mistral 7B
embedding model
reranker
Whisper
```

You could run:

```text
                  API
                   │
       ┌───────────┼───────────┐
       ↓           ↓           ↓
     LLM       Embedding    Whisper
    server       server      server
```

Your application decides which model to call.

This is very similar to the architecture you're eventually interested in for your local AI assistant.

---

# 62. Model routing

You can even have:

```text
User request
     ↓
Router
     │
     ├── coding → coding model
     ├── document question → RAG model
     ├── speech → Whisper
     └── image → vision model
```

This is often more useful than trying to force one model to do everything.

---

# 63. RAG + inference serving

This connects your previous topic directly.

Suppose you ask:

> "What does my college syllabus say about Compiler Design?"

Your architecture could be:

```text
Question
   ↓
Embedding model
   ↓
Vector database
   ↓
Retrieve relevant chunks
   ↓
Context
   ↓
Prompt
   ↓
LLM inference server
   ↓
Answer
```

The LLM itself doesn't need to contain your syllabus in its weights.

The inference server simply generates an answer using the supplied context.

---

# 64. Agent + inference serving

For your eventual coding agent:

```text
User
 ↓
Agent
 ↓
LLM server
 ↓
Decision
 ↓
Tool
 ↓
Result
 ↓
LLM server
 ↓
Decision
 ↓
...
```

For example:

```text
User:
"Fix this Docker problem."

Agent
 ↓
LLM
 ↓
run terminal
 ↓
read error
 ↓
LLM
 ↓
modify Dockerfile
 ↓
run test
 ↓
LLM
 ↓
answer
```

The inference server becomes the **reasoning/generation engine** of the agent.

---

# 65. Inference is not the same as RAG

This distinction is extremely important.

### Inference

```text
Prompt
 ↓
Model
 ↓
Answer
```

### RAG

```text
Question
 ↓
Retrieve information
 ↓
Context
 ↓
Prompt
 ↓
Model inference
 ↓
Answer
```

RAG uses inference.

Inference does not require RAG.

---

# 66. Inference is not fine-tuning

### Fine-tuning

Changes model parameters:

```text
Dataset
 ↓
Training
 ↓
Updated weights
```

### Inference

Uses the parameters:

```text
Prompt
 ↓
Fixed weights
 ↓
Output
```

---

# 67. Inference is not evaluation

### Evaluation

Measures model performance:

```text
Dataset
 ↓
Model
 ↓
Predictions
 ↓
Metrics
```

### Inference

Simply generates predictions.

Evaluation usually **uses inference** to obtain those predictions.

---

# 68. The entire LLM lifecycle

Now you can see the whole picture:

```text
DATASET
   ↓
DATA CLEANING
   ↓
PRETRAINING / FINE-TUNING
   ↓
MODEL
   ↓
EVALUATION
   ↓
QUANTIZATION / OPTIMIZATION
   ↓
INFERENCE
   ↓
SERVING
   ↓
APPLICATION
```

And for RAG:

```text
Documents
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector DB
 ↓
Retriever
 ↓
Context
 ↓
LLM inference
 ↓
Answer
```

---

# 69. The practical Hugging Face workflow

A realistic beginner workflow:

```text
1. Choose model
       ↓
2. Download model
       ↓
3. Load tokenizer
       ↓
4. Load model
       ↓
5. Move to GPU/CPU
       ↓
6. Tokenize prompt
       ↓
7. Generate
       ↓
8. Decode
       ↓
9. Measure speed/memory
       ↓
10. Optimize
       ↓
11. Wrap in API
       ↓
12. Dockerize
       ↓
13. Deploy
```

---

# 70. Your first practical inference program

A simplified example:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "your-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

prompt = "Explain Docker in simple terms."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inputs = {
    k: v.to(model.device)
    for k, v in inputs.items()
}

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(response)
```

Don't worry about memorizing this.

Understand the flow.

```text
load tokenizer
      ↓
load model
      ↓
prepare input
      ↓
inference mode
      ↓
generate
      ↓
decode
```

---

# 71. One important issue with that example

Notice:

```python
temperature=0.7
```

doesn't necessarily do anything useful if:

```python
do_sample=False
```

because temperature is relevant to sampling.

So for sampling:

```python
do_sample=True
temperature=0.7
```

For greedy generation:

```python
do_sample=False
```

This kind of detail becomes important when you're actually writing inference code.

---

# 72. A practical API

Once your local inference works:

```text
model.py
```

can become:

```text
server.py
```

Conceptually:

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/generate")
def generate(prompt: str):

    response = generate_with_model(prompt)

    return {
        "response": response
    }
```

Then:

```text
Flutter
  ↓
HTTP POST
  ↓
FastAPI
  ↓
Model
  ↓
Response
```

Congratulations — you now have a basic LLM service.

---

# 73. Streaming API architecture

A better chatbot architecture:

```text
Flutter
   ↓
POST /chat
   ↓
Inference server
   ↓
Generate token
   ↓
STREAM
   ↓
Flutter displays token
   ↓
Generate next token
   ↓
STREAM
   ↓
...
```

This creates the familiar ChatGPT-like typing effect.

---

# 74. What should you measure?

When experimenting with inference, don't just say:

> "It feels fast."

Measure it.

Record:

```text
Model:
Quantization:
GPU:
VRAM:
Context:
Input tokens:
Output tokens:

TTFT:
Tokens/sec:
Total latency:
Peak VRAM:
```

For example:

```text
Model: 14B
Quantization: 4-bit
GPU: RTX 3060 Ti
Context: 4K

Input: 500 tokens
Output: 200 tokens

TTFT: 0.9 sec
Generation: 18 tok/s
Peak VRAM: 7.6 GB
```

Those measurements let you compare configurations objectively.

---

# 75. Important performance bottlenecks

If inference is slow, possible bottlenecks include:

```text
CPU
GPU compute
GPU memory bandwidth
VRAM capacity
CPU↔GPU transfers
KV cache
context length
batch size
model architecture
quantization
runtime implementation
```

Don't automatically assume:

> "My GPU is weak."

The bottleneck could be something else.

---

# 76. GPU utilization

You might run:

```bash
nvidia-smi
```

and see:

```text
GPU utilization: 40%
```

That doesn't necessarily mean your model is performing badly.

You need to understand what the workload is doing.

For example:

```text
small batch
+
memory-bound workload
```

can behave differently from:

```text
large batch
+
compute-heavy workload
```

---

# 77. Memory bandwidth

LLM inference can be heavily affected by memory bandwidth because large amounts of model weights need to be accessed.

This is one reason GPU architecture matters beyond simply:

```text
number of CUDA cores
```

For local LLM inference, you should pay attention to:

```text
VRAM capacity
memory bandwidth
compute capability
quantization support
software/runtime support
```

---

# 78. Why bigger models aren't automatically better for your computer

Suppose:

```text
7B → 30 tok/s
```

but:

```text
30B → 5 tok/s
```

The 30B model may have capabilities you want, but the practical experience can be much slower.

Therefore model selection involves:

```text
quality
+
speed
+
memory
+
context
+
task requirements
```

not merely parameter count.

---

# 79. Model serving at scale

Imagine:

```text
1 user
```

Easy.

Now:

```text
100 users
```

You need:

```text
queueing
batching
scheduling
multiple GPUs
load balancing
```

At much larger scale:

```text
Load Balancer
      ↓
 ┌────┼────┐
 ↓    ↓    ↓
GPU1 GPU2 GPU3
```

Each worker can serve requests.

---

# 80. GPU parallelism

Large models may require multiple GPUs.

For example:

```text
GPU 1
 ↓
layers 1–20

GPU 2
 ↓
layers 21–40
```

This is model parallelism.

There are several kinds of parallelism, including:

```text
data parallelism
tensor parallelism
pipeline parallelism
```

You don't need to master these for your first local model.

But you should know they exist.

---

# 81. Quantization vs distillation

Don't confuse these.

### Quantization

Represent the **same model's weights** with lower precision.

```text
FP16 → INT4
```

### Distillation

Train a smaller model to imitate another model.

```text
Large teacher
      ↓
Student training
      ↓
Smaller model
```

Both can make deployment easier, but they are different techniques.

---

# 82. Model compilation

Inference frameworks may optimize models using:

```text
CUDA kernels
TensorRT
torch.compile
specialized kernels
Flash Attention
```

The goal is essentially:

```text
same model
+
better execution
=
faster inference
```

But compatibility varies by model and hardware.

---

# 83. FlashAttention

You've learned attention.

Now connect it to inference optimization.

Standard attention can require substantial memory.

FlashAttention uses an optimized implementation that reduces memory traffic and improves efficiency on supported hardware.

Conceptually:

```text
Normal attention
     ↓
large intermediate memory operations

FlashAttention
     ↓
optimized GPU computation
```

You don't need to implement it from scratch to benefit from it.

Modern libraries can use optimized attention kernels automatically or through configuration.

---

# 84. The most important distinction: model vs runtime vs server

This is a great mental model.

### Model

The neural network:

```text
Qwen
Llama
Mistral
...
```

### Runtime / inference engine

Software that executes it efficiently:

```text
Transformers
vLLM
llama.cpp
...
```

### Server

Makes it accessible:

```text
HTTP API
streaming
authentication
request handling
...
```

So:

```text
MODEL
  ↓
INFERENCE ENGINE
  ↓
SERVER
  ↓
APPLICATION
```

Sometimes one tool combines multiple layers.

---

# 85. Example: Ollama

With Ollama, many details are bundled together:

```text
Model management
+
runtime
+
local serving
+
API
```

So you can interact with:

```text
Your application
       ↓
Ollama API
       ↓
Model
       ↓
GPU/CPU
```

That's why Ollama feels much simpler than manually building every component.

---

# 86. Example: vLLM

A conceptual vLLM architecture:

```text
Client
  ↓
HTTP API
  ↓
vLLM
  ↓
Scheduler
  ↓
Continuous batching
  ↓
KV cache management
  ↓
GPU
  ↓
Model
```

The important thing is that vLLM isn't simply:

```text
"load model and call generate"
```

It's an inference-serving system designed around efficient concurrent generation.

---

# 87. Example: llama.cpp

A conceptual architecture:

```text
GGUF model
    ↓
llama.cpp
    ↓
CPU / GPU / hybrid
    ↓
generation
```

This is extremely useful for local deployment, particularly when memory is limited.

---

# 88. What you should learn practically

Given your goals, I'd learn inference in this order:

### Level 1 — Basic inference

Learn:

```text
AutoTokenizer
AutoModelForCausalLM
generate()
decode()
```

Understand:

```text
tokens
logits
generation
```

---

### Level 2 — Generation

Learn:

```text
max_new_tokens
temperature
top-k
top-p
sampling
greedy decoding
stop sequences
```

---

### Level 3 — Performance

Learn:

```text
FP32
FP16
BF16
INT8
INT4
VRAM
KV cache
context length
batching
tokens/sec
TTFT
```

---

### Level 4 — Serving

Learn:

```text
FastAPI
REST API
SSE streaming
OpenAI-compatible APIs
Docker
```

---

### Level 5 — Production inference

Learn:

```text
vLLM
TGI
llama.cpp
Ollama
continuous batching
PagedAttention
GPU scheduling
monitoring
```

---

# 89. Your first serious project

I strongly recommend building this:

## Local LLM API

```text
                 ┌───────────────┐
                 │ Flutter/Web   │
                 └───────┬───────┘
                         │
                         ↓
                 ┌───────────────┐
                 │ FastAPI       │
                 │ /chat         │
                 └───────┬───────┘
                         │
                         ↓
                 ┌───────────────┐
                 │ LLM Runtime   │
                 └───────┬───────┘
                         │
                         ↓
                      GPU
```

Implement:

```text
POST /chat
POST /generate
GET /health
```

Then add:

```text
streaming
conversation history
token counting
logging
authentication
Docker
```

Then replace your basic Transformers inference with:

```text
vLLM
```

or another specialized runtime.

That progression teaches you far more than simply watching videos.

---

# 90. Then add RAG

Your next architecture becomes:

```text
                    User
                      ↓
                   Flutter
                      ↓
                  FastAPI
                      ↓
              ┌───────┴────────┐
              ↓                ↓
          Retriever          LLM
              ↓                ↑
         Vector DB             │
              ↓                │
        Relevant chunks ───────┘
                      ↓
                    Answer
```

Now you're combining:

```text
Embeddings
+
RAG
+
LLM
+
Inference
+
Serving
+
API
```

That is already a legitimate AI application architecture.

---

# 91. Then add your LoRA model

Eventually:

```text
Base model
     +
Coding LoRA
     ↓
Coding assistant
```

Your architecture:

```text
VS Code
   ↓
Your agent
   ↓
FastAPI
   ↓
Inference server
   ↓
Fine-tuned model
   ↓
GPU
```

And tools:

```text
Terminal
Files
Git
Web
Docker
Kubernetes
```

Now you're approaching the kind of local coding agent you've been describing.

---

# 92. The complete mental model

If you remember only one diagram from this explanation, remember this:

```text
                       ┌───────────────┐
                       │     USER      │
                       └───────┬───────┘
                               │
                               ↓
                       ┌───────────────┐
                       │ APPLICATION   │
                       │ Flutter/Web   │
                       └───────┬───────┘
                               │
                               ↓
                       ┌───────────────┐
                       │   API SERVER  │
                       │ FastAPI/etc.  │
                       └───────┬───────┘
                               │
                               ↓
                       ┌───────────────┐
                       │    RUNTIME    │
                       │ vLLM / HF /   │
                       │ llama.cpp     │
                       └───────┬───────┘
                               │
                               ↓
                       ┌───────────────┐
                       │     MODEL     │
                       │ Qwen/Llama/...│
                       └───────┬───────┘
                               │
                         ┌─────┴─────┐
                         ↓           ↓
                       CPU          GPU
                                    │
                                    ↓
                               KV CACHE
                                    │
                                    ↓
                            TOKEN GENERATION
                                    │
                                    ↓
                               TEXT OUTPUT
```

And the internal generation loop is:

```text
Prompt
  ↓
Tokenization
  ↓
Token IDs
  ↓
Embeddings
  ↓
Transformer
  ↓
Logits
  ↓
Sampling / decoding
  ↓
Next token
  ↓
KV cache
  ↓
Next token
  ↓
Next token
  ↓
...
  ↓
EOS / stop condition
  ↓
Decoded text
```

---

# 93. What you should be able to explain after learning this

You should now be able to answer:

**What is inference?**

> Running a trained model forward to produce predictions without updating its weights.

**What is serving?**

> Making inference available to applications/users through an interface such as an API.

**Why does an LLM generate one token at a time?**

> Causal LMs perform autoregressive next-token prediction.

**What is KV cache?**

> Stored attention K/V states from previous tokens that avoid recomputing them during generation.

**What is prefill?**

> Processing the input/context before autoregressive generation begins.

**What is decode?**

> The iterative generation of new tokens after prefill.

**What is quantization?**

> Representing model weights/operations with lower numerical precision to reduce memory and potentially improve efficiency.

**What is batching?**

> Processing multiple requests together to improve hardware utilization.

**What is continuous batching?**

> Dynamically scheduling active requests together as requests enter and finish.

**What is streaming?**

> Sending generated output incrementally rather than waiting for the entire response.

**What is vLLM?**

> A high-performance LLM inference/serving engine designed for efficient generation and concurrent workloads.

**What is Ollama?**

> A convenient local model runtime/serving solution that packages much of the complexity of running models locally.

**What is llama.cpp?**

> An efficient inference ecosystem particularly associated with local/CPU/GPU-hybrid execution and GGUF models.

---

# 94. Where inference fits in your overall LLM roadmap

You've essentially been building the stack in the right direction:

```text
                    LLM DEVELOPMENT STACK

                 ┌─────────────────────┐
                 │     APPLICATION     │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │   Agents / RAG      │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ Inference / Serving │  ← YOU ARE HERE
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ Evaluation          │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ SFT / Fine-tuning   │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ LoRA / QLoRA        │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ Datasets / Cleaning │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ Transformer         │
                 │ Attention           │
                 │ Tokenization        │
                 │ Embeddings          │
                 └─────────────────────┘
```

The **next practical step** after understanding this theory is not another huge amount of theory. Build a tiny Hugging Face inference script, measure **tokens/sec + VRAM + latency**, then turn it into a **FastAPI streaming endpoint**, and finally run the same model through **Ollama/vLLM** so you can directly see what an inference engine/serving layer actually adds.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)