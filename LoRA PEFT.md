# LoRA PEFT

# LoRA / PEFT — from zero to practical

## 1. First: what problem are LoRA and PEFT solving?

Suppose you have a pretrained LLM:

> **Qwen / Llama / Mistral → billions of parameters → already knows language, coding, reasoning patterns, etc.**

You want to specialize it for a task:

- coding in a particular style
- medical/legal/technical text
- your organization's documentation
- instruction following
- a particular output format
- a specific domain
- a particular conversational style

The traditional approach is **full fine-tuning**.

You take the model's parameters:

$$
W
$$

and train **all of them**.

For a 7B model:

$$
7,000,000,000
$$

parameters may need to participate in training.

That's expensive.

### LoRA asks:

> **What if I don't change the original model at all, and instead learn a small set of additional parameters that modify its behavior?**

That's the fundamental idea.

---

# 2. PEFT vs LoRA

These two terms are related but aren't exactly the same thing.

### PEFT

**Parameter-Efficient Fine-Tuning**

is the **general category**.

It means:

> Fine-tune a pretrained model while updating only a small fraction of its parameters.

Methods include:

- LoRA
- AdaLoRA
- IA³
- Prefix Tuning
- Prompt Tuning
- adapters
- etc.

### LoRA

**Low-Rank Adaptation**

is one particular PEFT method.

So:

```text
PEFT
│
├── LoRA
├── AdaLoRA
├── Prefix Tuning
├── Prompt Tuning
├── IA³
└── other methods
```

When people say:

> "I'm doing PEFT"

they may mean LoRA or another PEFT technique.

When they say:

> "I'm doing LoRA"

they're talking about a specific PEFT algorithm.

---

# 3. Before LoRA: understand normal fine-tuning

Imagine a neural-network layer:

$$
y = Wx
$$

where:

- $x$ = input
- $W$ = learned weight matrix
- $y$ = output

Suppose:

$$
W =
\begin{bmatrix}
0.2 & 0.5 & 0.1\\
0.4 & 0.7 & 0.3\\
0.8 & 0.2 & 0.6
\end{bmatrix}
$$

During normal training, gradient descent changes $W$.

You start with:

$$
W_0
$$

and training learns:

$$
W'
$$

Therefore:

$$
W' = W_0 + \Delta W
$$

where:

$$
\Delta W
$$

is the change learned during fine-tuning.

This equation is extremely important:

$$
\boxed{W' = W + \Delta W}
$$

Full fine-tuning learns the entire:

$$
\Delta W
$$

matrix.

---

# 4. The key observation behind LoRA

LoRA says:

> We probably don't need a completely arbitrary huge $\Delta W$.

Instead, perhaps the useful change can be represented using two **small matrices**.

Instead of learning:

$$
\Delta W
$$

directly, LoRA represents it as:

$$
\boxed{\Delta W = BA}
$$

Therefore:

$$
\boxed{W' = W + BA}
$$

where:

- $W$ = original pretrained weights
- $A$ = small trainable matrix
- $B$ = small trainable matrix

This is LoRA.

---

# 5. What does "low-rank" mean?

This is the part that initially sounds scary but is actually straightforward.

Suppose your original weight matrix is:

$$
W \in \mathbb{R}^{10000 \times 10000}
$$

That's:

$$
100,000,000
$$

parameters.

Instead of learning a full:

$$
10000 \times 10000
$$

update, LoRA chooses a small rank:

$$
r = 8
$$

and creates:

$$
A \in \mathbb{R}^{8 \times 10000}
$$

and:

$$
B \in \mathbb{R}^{10000 \times 8}
$$

Then:

$$
BA
$$

has dimensions:

$$
10000 \times 8
$$

times

$$
8 \times 10000
$$

giving:

$$
10000 \times 10000
$$

So the final update has the same shape as $W$:

$$
\Delta W = BA
$$

but it is represented using far fewer trainable parameters.

---

# 6. How many parameters did we save?

Full fine-tuning:

$$
10000 \times 10000
=
100,000,000
$$

LoRA:

$$
(10000 \times 8)+(8 \times 10000)
$$

$$
=80,000+80,000
$$

$$
=160,000
$$

So:

```text
Full fine-tuning
100,000,000 parameters

LoRA
160,000 parameters
```

That's only:

$$
0.16\%
$$

of the original parameters.

This is the central reason LoRA is useful.

---

# 7. How LoRA actually changes the computation

Normal layer:

$$
y = Wx
$$

LoRA:

$$
\boxed{y = Wx + BAx}
$$

You can rearrange:

$$
y = Wx + B(Ax)
$$

So practically:

```text
                  ┌───────────────┐
x ───────────────►│ Original W    │──────►
│                 └───────────────┘
│
│                 ┌─────┐   ┌─────┐
└────────────────►│  A  │──►│  B  │──────►
                  └─────┘   └─────┘

                     ↓
                 LoRA update

                    ↓
                  ADD

                    ↓

                    y
```

The original model remains available.

LoRA adds a learned correction.

---

# 8. Why is the original model frozen?

This is one of the most important practical concepts.

During LoRA training:

```text
Original model weights
        ↓
      FROZEN
        ↓
No gradient updates

LoRA matrices
        ↓
      TRAINABLE
        ↓
Gradient updates
```

So if your model has:

**7 billion parameters**

you might train only something like:

**a few million parameters** depending on the architecture and LoRA configuration.

The exact percentage depends heavily on:

- model architecture
- target modules
- LoRA rank
- number of layers
- whether biases are trained

---

# 9. Why does such a tiny number of parameters work?

Because the pretrained model already contains a tremendous amount of knowledge.

Imagine you have a programmer who already knows:

- Python
- C
- C++
- algorithms
- databases
- Linux
- networking
- machine learning

You don't need to teach programming from zero.

You might only need to teach:

> "When answering users, follow this particular format."

or:

> "Generate code according to this organization's conventions."

LoRA tries to learn the **adaptation** rather than relearning everything.

So:

```text
Pretrained model
      +
small learned adaptation
      ↓
specialized model
```

---

# 10. LoRA in an LLM

Now connect this to the Transformer you learned earlier.

A Transformer contains many linear projections.

For example, attention contains:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

and the attention output has another projection:

$$
O = XW_O
$$

LoRA can modify these weight matrices.

For example:

$$
W_Q' = W_Q + B_QA_Q
$$

and:

$$
W_V' = W_V + B_VA_V
$$

So:

$$
Q = X(W_Q+B_QA_Q)
$$

$$
V = X(W_V+B_VA_V)
$$

You don't necessarily have to LoRA every layer.

---

# 11. Where is LoRA usually applied?

Common targets include attention projections:

```text
q_proj
k_proj
v_proj
o_proj
```

and sometimes MLP projections:

```text
gate_proj
up_proj
down_proj
```

A common beginner configuration might target:

```text
q_proj
v_proj
```

Another configuration may target:

```text
q_proj
k_proj
v_proj
o_proj
```

or attention + MLP layers.

There isn't one universally correct configuration.

---

# 12. Why Q/K/V matters here

You recently learned Q/K/V.

Now you can connect the concepts.

Attention:

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

Suppose you want your model to become better at a particular domain.

LoRA can learn:

$$
W_Q' = W_Q+\Delta W_Q
$$

$$
W_K' = W_K+\Delta W_K
$$

$$
W_V' = W_V+\Delta W_V
$$

So you're effectively teaching the Transformer:

> "Use its existing representation and attention mechanisms slightly differently for this task."

That's why understanding Q/K/V makes LoRA much easier.

---

# 13. LoRA rank $r$

One of the most important hyperparameters is:

$$
\boxed{r}
$$

called the **rank**.

Examples:

```text
r = 4
r = 8
r = 16
r = 32
r = 64
```

Smaller $r$:

```text
fewer trainable parameters
less memory
less capacity
```

Larger $r$:

```text
more trainable parameters
more memory
more adaptation capacity
```

But:

> Higher rank does NOT automatically mean better.

You choose it based on the task, model, dataset, and compute.

---

# 14. LoRA scaling: alpha

You will often see:

```python
r=16
lora_alpha=32
```

The LoRA update is commonly scaled using:

$$
\frac{\alpha}{r}
$$

So conceptually:

$$
W' = W + \frac{\alpha}{r}BA
$$

If:

$$
r=16
$$

and:

$$
\alpha=32
$$

then:

$$
\frac{\alpha}{r}=2
$$

The exact implementation details can vary by LoRA variant/configuration, but this is the basic idea you need.

---

# 15. What is LoRA dropout?

You may encounter:

```python
lora_dropout=0.05
```

It applies dropout to the LoRA pathway during training.

Conceptually:

```text
Input
  │
  ├──────────────► Original model
  │
  └──► LoRA ──► dropout ──► LoRA update
```

It's a regularization mechanism.

Typical values might be:

```text
0
0.05
0.1
```

depending on the task.

---

# 16. What happens at initialization?

This is a beautiful detail.

Initially, you want the LoRA adapter to make essentially **no change** to the pretrained model.

So LoRA is initialized such that:

$$
BA \approx 0
$$

Therefore:

$$
W' \approx W
$$

At the beginning:

```text
Pretrained model
       +
almost-zero LoRA update
       ↓
almost the original model
```

Then training gradually learns the adaptation.

---

# 17. Full fine-tuning vs LoRA

Here's the conceptual difference:

| | Full fine-tuning | LoRA |
|---|---|---|
| Original weights | Updated | Frozen |
| Trainable parameters | Almost/all | Small fraction |
| GPU memory | High | Much lower |
| Training cost | High | Lower |
| Storage per specialization | Large | Small adapter |
| Multiple specializations | Expensive | Easy |
| Base model required | Usually stored as trained model | Base + adapter |

Imagine you have:

```text
Qwen 7B
```

You could create:

```text
Qwen-7B-medical
Qwen-7B-coding
Qwen-7B-legal
Qwen-7B-instruction
```

with full fine-tuning.

That's expensive.

With LoRA:

```text
Qwen 7B
│
├── coding_adapter
├── medical_adapter
├── legal_adapter
└── instruction_adapter
```

The base model can remain the same.

---

# 18. This is extremely useful for your local-LLM plan

You previously wanted something like:

```text
Base LLM
     │
     ├── Coding LoRA
     │
     ├── DevOps/Web LoRA
     │
     └── College-specific adaptation
```

That's exactly the kind of architecture where LoRA becomes useful.

For example:

```text
Qwen 14B
     │
     ├──────── coding_adapter
     │
     ├──────── devops_adapter
     │
     └──────── college_adapter
```

You can load the base model and activate the adapter relevant to the task.

But there's an important distinction:

> **LoRA is not a replacement for RAG.**

---

# 19. LoRA vs RAG

This distinction is critical.

Suppose you have 500 college PDFs.

You could put those documents into a vector database and use RAG.

```text
Question
   ↓
Embedding
   ↓
Vector database
   ↓
Relevant chunks
   ↓
LLM
   ↓
Answer
```

That's **RAG**.

LoRA is different.

```text
Training dataset
       ↓
    LoRA training
       ↓
   LoRA adapter
       ↓
Base model + adapter
```

RAG gives the model **external information at inference time**.

LoRA changes the model's **learned behavior/weights through training**.

---

# 20. Example: college PDFs

Suppose your PDFs say:

> "Semester 3 syllabus contains Operating Systems, DBMS..."

You generally don't need LoRA to make the model memorize the syllabus.

RAG is usually more appropriate:

```text
PDF
 ↓
chunk
 ↓
embedding
 ↓
vector DB
 ↓
retrieve
 ↓
LLM
```

Then the model can answer based on the current documents.

But suppose you want:

> "Always answer questions in this particular structured format."

or:

> "Generate explanations using this specific tutoring style."

LoRA may be useful.

---

# 21. LoRA is also not simply "memorization"

This is another common misunderstanding.

Suppose you train LoRA on:

```text
Question → Answer
Question → Answer
Question → Answer
...
```

The goal isn't merely:

> "Store these answers."

You're changing how the model maps inputs to outputs.

For example:

```text
Before:

User: Explain TCP
Model: [generic explanation]

After LoRA:

User: Explain TCP
Model: 
1. Definition
2. Packet structure
3. Connection establishment
4. Example
5. Common mistakes
```

The adaptation is behavioral.

---

# 22. Now let's understand PEFT practically

Hugging Face has libraries that make this relatively straightforward.

A typical modern ecosystem looks like:

```text
PyTorch
   ↓
Transformers
   ↓
PEFT
   ↓
LoRA
   ↓
Trainer / SFTTrainer
```

You don't manually calculate:

$$
BA
$$

for every layer.

The PEFT library handles the adapter machinery.

---

# 23. The typical training pipeline

Your practical workflow becomes:

```text
1. Choose base model
        ↓
2. Prepare dataset
        ↓
3. Format examples
        ↓
4. Tokenize
        ↓
5. Load model
        ↓
6. Create LoRA configuration
        ↓
7. Attach LoRA adapters
        ↓
8. Train only adapters
        ↓
9. Evaluate
        ↓
10. Save adapter
        ↓
11. Load base + adapter
        ↓
12. Test inference
```

This is the workflow you should understand.

---

# 24. Dataset for LoRA/SFT

For instruction fine-tuning, you might have:

```json
{
  "instruction": "Explain Docker volumes.",
  "input": "",
  "output": "A Docker volume is..."
}
```

Or conversational data:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explain Docker volumes."
    },
    {
      "role": "assistant",
      "content": "Docker volumes are..."
    }
  ]
}
```

The exact dataset format depends on the model and training framework.

---

# 25. The important thing: quality > quantity

A huge dataset isn't automatically good.

Suppose you have:

```text
1,000,000 terrible examples
```

versus:

```text
20,000 high-quality examples
```

The second may be much more useful.

Your training examples should ideally be:

- correct
- consistent
- relevant
- diverse
- properly formatted
- free from unnecessary duplication
- representative of the behavior you want

---

# 26. A practical LoRA example

Let's say you have:

**Qwen 7B**

and want to make it better at generating your preferred coding explanations.

Your data:

```text
User:
Explain linked-list insertion.

Assistant:
1. Create a new node.
2. Allocate memory.
3. Set the data.
4. Connect the node...
```

Thousands of examples like this.

Then:

```text
Qwen 7B
   │
   ├── frozen parameters
   │
   └── LoRA adapters
             ↓
         training
             ↓
    coding-style adapter
```

After training:

```text
Qwen 7B
+
coding LoRA
```

Now the adapter modifies the model's behavior.

---

# 27. What does training actually update?

This is extremely important.

Suppose:

```text
Base model = 7B parameters
```

During LoRA training:

```text
Base:
Wq ── frozen
Wk ── frozen
Wv ── frozen
Wo ── frozen
MLP ── frozen
...
```

LoRA:

```text
Aq ── trainable
Bq ── trainable

Av ── trainable
Bv ── trainable
...
```

Backpropagation computes gradients.

But only the trainable LoRA parameters are updated.

So:

$$
A \leftarrow A-\eta\nabla_A L
$$

$$
B \leftarrow B-\eta\nabla_B L
$$

while:

$$
W \text{ remains unchanged}
$$

---

# 28. Why LoRA reduces optimizer memory

This is one of its biggest practical benefits.

Training doesn't only need memory for model weights.

You also need memory for things like:

```text
model weights
gradients
optimizer states
activations
temporary tensors
```

For full fine-tuning, this can become enormous.

With LoRA:

```text
Base model
   ↓
mostly inference/frozen

LoRA parameters
   ↓
gradients
   ↓
optimizer states
```

Since far fewer parameters are trainable, optimizer and gradient memory can be dramatically reduced.

This is one reason a model that would be impractical to full-fine-tune may become feasible with LoRA.

---

# 29. LoRA + quantization = QLoRA

This is extremely important for you.

You'll often hear:

> **QLoRA**

QLoRA combines:

```text
Quantization
+
LoRA
```

The basic idea is:

```text
Large pretrained model
       ↓
quantized representation
       ↓
frozen base model
       +
LoRA adapters
       ↓
training
```

Instead of loading the base model in something like FP16/BF16, you can use a much lower-bit representation, commonly 4-bit in the QLoRA approach.

This reduces memory requirements substantially.

---

# 30. LoRA vs QLoRA

Think of them as:

### LoRA

```text
FP16/BF16 base model
+
LoRA
```

### QLoRA

```text
4-bit quantized base model
+
LoRA
```

The important point:

> **QLoRA doesn't mean the LoRA adapter itself is necessarily 4-bit.**

The base model is quantized; the trainable adapter typically remains in a higher precision suitable for training.

---

# 31. Why QLoRA matters for your RTX 3060 Ti

You have an **8 GB VRAM GPU**.

That changes what is practical.

Full fine-tuning of a modern 7B/14B model can be difficult or impossible within 8 GB VRAM.

LoRA helps.

QLoRA helps even more.

For example, conceptually:

```text
7B model

Full FT
████████████████████████████
Very large memory requirement

LoRA
████████████████
lower

QLoRA
████████
much lower base-model memory
```

Exact requirements depend heavily on:

- model architecture
- sequence length
- batch size
- gradient accumulation
- precision
- optimizer
- checkpointing
- LoRA configuration

So don't interpret "7B QLoRA" as having one universal VRAM requirement.

---

# 32. What is a LoRA adapter file?

This is one of the coolest practical aspects.

Suppose:

```text
Base model = 7B
```

Your trained LoRA might only be a small fraction of the base model's size.

You can save:

```text
adapter_config.json
adapter_model.safetensors
```

instead of saving another complete 7B model.

Conceptually:

```text
Base model
   +
adapter_model.safetensors
   +
adapter_config.json
```

gives you your specialized model.

---

# 33. Multiple adapters

This means you could have:

```text
models/
│
├── Qwen/
│
└── adapters/
    ├── coding/
    ├── devops/
    ├── college/
    └── documentation/
```

Then:

```text
Qwen + coding
```

or:

```text
Qwen + devops
```

or:

```text
Qwen + college
```

This is very convenient.

---

# 34. Adapter merging

You can also merge a LoRA adapter into the base model.

Conceptually:

```text
Base model
     +
LoRA adapter
     ↓
merged model
```

Mathematically:

$$
W_{\text{merged}}
=
W + \frac{\alpha}{r}BA
$$

Then you can use the resulting model without separately applying the adapter.

But keeping adapters separate is often useful because:

- easier experimentation
- smaller storage
- easy switching
- preserve the original base model
- multiple specializations

---

# 35. LoRA doesn't create a new foundation model

This is important for your Hugging Face goals.

Suppose you take:

```text
Qwen 7B
```

and train a coding LoRA.

You haven't trained a new 7B foundation model from scratch.

You've created:

> **a LoRA adapter that specializes Qwen 7B.**

That's a meaningful contribution, but you should represent it correctly when publishing it.

---

# 36. LoRA vs training from scratch

These are completely different scales.

### Pretraining

```text
raw internet/books/code/data
        ↓
massive compute
        ↓
foundation model
```

### SFT

```text
pretrained model
        ↓
instruction examples
        ↓
behavior specialization
```

### LoRA-SFT

```text
pretrained model
        ↓
freeze base
        ↓
train small LoRA adapters
        ↓
specialized adapter
```

You will mostly be working in the third category when experimenting on your hardware.

---

# 37. LoRA vs SFT

This distinction is **very important**.

They're not competing concepts.

**SFT** describes the **training objective/process**:

> supervised fine-tuning using examples of desired behavior.

**LoRA** describes **how the model parameters are adapted**.

So you can have:

```text
SFT + Full Fine-Tuning
```

or:

```text
SFT + LoRA
```

or:

```text
SFT + QLoRA
```

Think:

```text
WHAT are you doing?
        ↓
       SFT

HOW are you updating the model?
        ↓
       LoRA
```

This mental model will save you a lot of confusion.

---

# 38. A complete picture

Put everything you've been learning together:

```text
                    PRETRAINED LLM
                         │
                         │
                    Transformer
                         │
            ┌────────────┴────────────┐
            │                         │
       Attention                     MLP
            │
       Q / K / V
            │
       Linear layers
            │
            ▼
       LoRA attaches
            │
      ┌─────┴─────┐
      │           │
      A           B
      │           │
      └─────┬─────┘
            │
         BA update
            │
            ▼
      Adapted model
```

And the training process:

```text
Dataset
   ↓
Tokenization
   ↓
Input IDs
   ↓
Transformer
   ↓
Frozen base weights
   +
Trainable LoRA weights
   ↓
Forward pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Update LoRA A/B
   ↓
Repeat
```

---

# 39. A practical Hugging Face implementation

Now let's move from theory to code.

A simplified setup looks like this:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()
```

You might see something conceptually like:

```text
trainable params: 8M
all params: 7B
trainable%: 0.1%
```

The exact numbers depend on the model and target modules.

---

# 40. What each setting means

### `r`

```python
r=16
```

LoRA rank.

Higher:

```text
more capacity
more trainable parameters
```

Lower:

```text
less capacity
fewer trainable parameters
```

---

### `lora_alpha`

```python
lora_alpha=32
```

Controls the scaling of the LoRA update.

Commonly the effective scaling relates to:

$$
\frac{\alpha}{r}
$$

---

### `lora_dropout`

```python
lora_dropout=0.05
```

Regularization applied to the LoRA pathway during training.

---

### `target_modules`

```python
target_modules=["q_proj", "v_proj"]
```

This tells PEFT:

> Attach LoRA to these model layers.

This is model-architecture dependent.

**Do not blindly copy `q_proj`/`v_proj` to every model.**

For a different architecture, the layer names may differ.

---

### `task_type`

```python
task_type="CAUSAL_LM"
```

You're telling PEFT that you're adapting a causal language model.

---

# 41. Always inspect trainable parameters

This is one of the first things you should do after creating a LoRA model.

```python
model.print_trainable_parameters()
```

Why?

Because you want to verify:

> "Am I actually training only the adapter?"

If something goes wrong and millions/billions of base parameters become trainable, you'll know immediately.

---

# 42. Using LoRA with SFTTrainer

A practical modern workflow can look approximately like:

```python
from peft import LoraConfig
from trl import SFTTrainer

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    ...
)

trainer.train()
```

The exact API changes across versions, so when you actually implement this, check the current documentation for the installed versions of **Transformers, PEFT, TRL, and bitsandbytes**.

---

# 43. What happens during training?

Suppose your training example is:

```text
User:
What is Docker?

Assistant:
Docker is a platform...
```

The tokenizer converts it into token IDs:

```text
[1514, 8273, 392, ...]
```

The model predicts the next tokens.

For causal LM:

$$
P(x_t|x_1,\ldots,x_{t-1})
$$

The model produces logits.

Then you calculate cross-entropy loss.

Something like:

```text
input
 ↓
Qwen
 ↓
logits
 ↓
loss
 ↓
backpropagation
 ↓
LoRA A/B gradients
 ↓
optimizer
 ↓
updated A/B
```

The base model remains frozen.

---

# 44. Why training can still be expensive

This is an important misconception.

Someone might think:

> "Only 0.1% of parameters are trainable, so training should be almost free."

Not necessarily.

You still have to run the **entire model forward pass**.

For every example:

```text
input
 ↓
all Transformer layers
 ↓
attention
 ↓
MLP
 ↓
output
```

The base model still consumes compute and memory.

LoRA primarily reduces:

- trainable parameters
- gradient storage
- optimizer state
- checkpoint size

It does **not** magically make the model's forward computation disappear.

---

# 45. Sequence length matters enormously

Suppose:

```text
sequence length = 512
```

versus:

```text
sequence length = 8192
```

The latter can dramatically increase memory and compute requirements.

This matters particularly for your projects involving:

- PDFs
- code
- documentation
- long-context data

Don't assume:

> "I have an 8 GB GPU, therefore LoRA training a model of size X will work."

You need to consider the entire training configuration.

---

# 46. Gradient accumulation

Your GPU might only fit:

```text
batch_size = 1
```

But you may want an effective batch size of:

```text
16
```

You can use gradient accumulation.

Conceptually:

```text
batch 1 → loss → accumulate
batch 2 → loss → accumulate
batch 3 → loss → accumulate
...
batch 16
      ↓
optimizer update
```

This allows a larger **effective batch size** without requiring all samples simultaneously in GPU memory.

---

# 47. Gradient checkpointing

Another important memory-saving technique.

Normally you store many intermediate activations:

```text
Layer 1 → activation
Layer 2 → activation
Layer 3 → activation
...
```

These consume memory.

Gradient checkpointing saves fewer activations and recomputes some during backward propagation.

Tradeoff:

```text
less memory
+
more computation
```

For a small-VRAM GPU, this can be useful.

---

# 48. Mixed precision

You will also encounter:

```text
FP32
FP16
BF16
```

Training modern LLMs commonly uses lower precision where supported.

For example:

```text
FP32 → high memory
FP16 → lower memory
BF16 → lower memory
```

The best choice depends on your GPU and software stack.

---

# 49. QLoRA practical architecture

A typical QLoRA setup might conceptually be:

```text
             Training dataset
                    │
                    ▼
               tokenizer
                    │
                    ▼
          ┌──────────────────┐
          │ Quantized LLM    │
          │                  │
          │  frozen 4-bit    │
          │     weights      │
          └────────┬─────────┘
                   │
              LoRA adapters
                   │
             trainable FP
                   │
                   ▼
               loss
                   │
              backprop
                   │
                   ▼
              update LoRA
```

That's the setup you should eventually experiment with on limited VRAM.

---

# 50. A practical project for you

Rather than immediately attempting:

> "Fine-tune Qwen 14B on a giant dataset."

I'd recommend learning LoRA using a small model first.

For example:

```text
small open model
       ↓
small high-quality dataset
       ↓
LoRA
       ↓
evaluate
       ↓
inspect outputs
```

Once the pipeline works:

```text
small model
     ↓
7B model
     ↓
QLoRA
     ↓
larger experiments
```

This teaches you much more efficiently.

---

# 51. Your first LoRA project

A very good beginner project would be:

### "Coding explanation adapter"

Dataset:

```text
Question → high-quality explanation
```

Examples:

```text
Explain recursion in C.

Explain pointers in C.

Explain linked-list insertion.

Explain Docker containers.

Explain Kubernetes Pods.
```

But don't mix everything randomly.

A better first project:

> **C programming explanation style LoRA**

Dataset:

```text
C question
    ↓
structured explanation
    ↓
example
    ↓
common mistake
    ↓
correct code
```

Then compare:

```text
Base model
vs
Base model + LoRA
```

This gives you a measurable before/after experiment.

---

# 52. Evaluation is extremely important

Don't just train and say:

> "It works."

Create a test set.

For example:

```text
Training:
800 examples

Validation:
100 examples

Test:
100 examples
```

Never train on your test examples.

Then compare:

```text
Base model:
response A

LoRA:
response B
```

Evaluate things like:

- correctness
- formatting
- instruction following
- hallucination
- consistency
- coding correctness

For coding, actually run generated code where safe.

---

# 53. Overfitting

LoRA can overfit.

Suppose you train too much on:

```text
small dataset
```

The adapter can become overly specialized.

You might observe:

```text
Training loss ↓↓↓

Validation performance:
   ↓
   ↓
   ↑ bad
```

This is a warning sign.

Possible causes include:

- too many epochs
- tiny dataset
- duplicated examples
- excessive learning rate
- overly large LoRA capacity
- poor dataset quality

---

# 54. Learning rate

This controls how aggressively LoRA parameters are updated.

Conceptually:

$$
\theta_{new}
=
\theta_{old}
-
\eta\nabla L
$$

where:

$$
\eta
$$

is the learning rate.

Because you're only training adapters, LoRA learning rates are often different from those used in full fine-tuning.

You should not blindly copy a learning rate from a completely different model/task.

---

# 55. Epoch

One epoch means:

> The training process has gone through the training dataset once.

For example:

```text
1,000 examples

1 epoch → 1,000 examples processed
3 epochs → 3,000 example passes
```

More epochs don't automatically mean better.

---

# 56. LoRA rank and learning rate interact with capacity

Think of LoRA rank as roughly controlling:

> **How much freedom does the adapter have to change the model?**

And learning rate as:

> **How quickly does it learn those changes?**

And dataset size/quality as:

> **What information is it learning from?**

So:

```text
Rank
 ↓
adapter capacity

Learning rate
 ↓
update magnitude/speed

Dataset
 ↓
information being learned

Epochs
 ↓
how many times it sees that information
```

All of these matter.

---

# 57. What exactly does "rank" mathematically mean?

A matrix's rank is the number of independent directions/information dimensions needed to represent it.

A full matrix:

$$
\Delta W
$$

may have very high rank.

LoRA constrains:

$$
\Delta W = BA
$$

so:

$$
rank(\Delta W)\leq r
$$

This is where the name **Low-Rank Adaptation** comes from.

If:

$$
r=8
$$

then:

$$
rank(BA)\leq8
$$

So instead of allowing an arbitrary update, we're restricting the update to a lower-dimensional structure.

---

# 58. The intuition behind low-rank adaptation

Imagine a huge 10,000-dimensional space.

You might think you need to modify the model in all 10,000 directions.

But perhaps the task mainly requires changes along a much smaller number of directions.

LoRA says:

> "Let's learn those important directions instead of modifying the entire parameter space."

This is the intuition behind the technique.

---

# 59. One subtle but important point

LoRA does **not** mean:

> "The model only learns r things."

No.

Suppose:

$$
W \in \mathbb{R}^{10000\times10000}
$$

and:

$$
r=8
$$

The resulting update:

$$
BA
$$

still has:

$$
10000\times10000
$$

shape.

It is just a **rank-constrained update** represented efficiently by two smaller matrices.

This distinction is important.

---

# 60. LoRA and catastrophic forgetting

Full fine-tuning can potentially cause the model's original capabilities to change or degrade, especially with small or narrow datasets.

LoRA helps because the base model stays unchanged.

You can simply remove the adapter:

```text
Base model
```

and recover the original behavior.

Then:

```text
Base + coding LoRA
```

gives specialized behavior.

This makes experimentation much safer and easier.

It doesn't guarantee that the adapted model won't have undesirable behavior, but it preserves the original base weights separately.

---

# 61. Can you combine LoRA with RAG?

Absolutely.

In fact, your eventual system could look like:

```text
                 User
                   │
                   ▼
             Local LLM
                   │
        ┌──────────┴──────────┐
        │                     │
   LoRA adapter             RAG
        │                     │
 behavior/style          knowledge
        │                     │
        └──────────┬──────────┘
                   ▼
                Answer
```

For example:

### LoRA

Teach:

> "How should the assistant behave?"

### RAG

Provide:

> "What information should the assistant use?"

That's a very useful mental distinction.

---

# 62. Your planned coding agent

Your earlier idea of:

```text
Local LLM
+
RAG
+
coding LoRA
+
DevOps RAG
+
tools
+
VS Code
```

can therefore be thought of as:

```text
                    Local LLM
                       │
        ┌──────────────┼──────────────┐
        │              │              │
       LoRA            RAG           Tools
        │              │              │
 behavior/style     knowledge     execute actions
        │              │              │
        └──────────────┼──────────────┘
                       │
                  Agent behavior
```

LoRA is only one component.

---

# 63. LoRA is not the same as prompting

Compare:

### Prompting

```text
"Always answer using numbered steps."
```

No training.

### RAG

```text
Retrieve relevant documentation.
```

No modification to model weights.

### LoRA

```text
Train adapter to produce the desired behavior.
```

Model adaptation.

### Full fine-tuning

```text
Train most/all model parameters.
```

Major model modification.

---

# 64. LoRA is also not the same as embeddings

Embeddings:

```text
text
 ↓
vector
```

used for:

- semantic search
- retrieval
- similarity
- clustering

LoRA:

```text
model
 ↓
trainable low-rank adapters
```

used for:

- model adaptation
- behavioral specialization
- domain/task adaptation

Completely different mechanisms.

---

# 65. The complete LLM toolbox

You can now categorize the concepts you've been learning:

```text
TOKENIZATION
"What pieces of text do I process?"

        ↓

EMBEDDINGS
"How do I represent tokens as vectors?"

        ↓

TRANSFORMER
"How do I process contextual information?"

        ↓

ATTENTION / QKV
"What should each token pay attention to?"

        ↓

CAUSAL LM
"Predict the next token."

        ↓

SFT
"Teach desired behavior using examples."

        ↓

LoRA / PEFT
"Do that adaptation efficiently."

        ↓

QLoRA
"Do LoRA while keeping the base model
quantized to reduce memory."

        ↓

RAG
"Give the model external knowledge at inference."

        ↓

TOOLS / AGENTS
"Let the model interact with the outside world."
```

That is a very useful roadmap.

---

# 66. A concrete mental model to remember

If you remember only one equation, remember:

$$
\boxed{W' = W + \Delta W}
$$

Full fine-tuning:

$$
\boxed{W' = W + \Delta W}
$$

where you directly learn a huge $\Delta W$.

LoRA:

$$
\boxed{W' = W + BA}
$$

or commonly with scaling:

$$
\boxed{
W' = W + \frac{\alpha}{r}BA
}
$$

where $A$ and $B$ are small trainable matrices.

And:

```text
W = frozen pretrained knowledge

A + B = trainable adaptation
```

That is LoRA.

---

# 67. The practical LoRA workflow you should memorize

When you eventually do this yourself:

```text
             1. Pick base model
                     ↓
             2. Pick dataset
                     ↓
             3. Clean dataset
                     ↓
             4. Format conversations
                     ↓
             5. Tokenize
                     ↓
             6. Load model
                     ↓
             7. Configure LoRA
                     ↓
             8. Attach adapters
                     ↓
             9. Verify trainable params
                     ↓
            10. Train
                     ↓
            11. Evaluate
                     ↓
            12. Save adapter
                     ↓
            13. Load base + adapter
                     ↓
            14. Test against base model
                     ↓
            15. Publish adapter
```

---

# 68. What you should learn next

Given your goal of eventually working with Hugging Face and building your own local LLM systems, I'd learn these in this order:

```text
                 LLM FOUNDATION
                       │
                       ▼
              Transformer architecture
                       │
                       ▼
                   Tokenization
                       │
                       ▼
                    Embeddings
                       │
                       ▼
                  Attention QKV
                       │
                       ▼
                Causal LM / LM loss
                       │
                       ▼
                       SFT
                       │
                       ▼
                 LoRA / PEFT   ← YOU ARE HERE
                       │
                       ▼
                     QLoRA
                       │
                       ▼
                Dataset creation
                       │
                       ▼
                 Evaluation
                       │
                       ▼
                     RAG
                       │
                       ▼
                 Tool calling
                       │
                       ▼
                    Agents
                       │
                       ▼
              Hugging Face ecosystem
```

And after this, the practical stack worth becoming comfortable with is roughly:

```text
Python
 ↓
PyTorch
 ↓
Hugging Face Transformers
 ↓
Datasets
 ↓
Tokenizers
 ↓
PEFT
 ↓
TRL / SFTTrainer
 ↓
bitsandbytes
 ↓
Accelerate
 ↓
Evaluation tools
 ↓
RAG / vector databases
 ↓
Agents / tool calling
```

---

# 69. Your first practical exercise

Once you understand the theory, don't immediately train a huge model.

Build this:

### Project 1 — LoRA from scratch conceptually

Take a tiny linear layer:

$$
y=Wx
$$

Implement:

```python
y = W @ x + B @ A @ x
```

Then freeze `W` and train only `A` and `B`.

This makes the LoRA concept completely concrete.

Then:

### Project 2

Use:

```text
small Hugging Face causal LM
+
PEFT
+
LoRA
+
small dataset
```

Train it.

Then:

### Project 3

Use:

```text
7B model
+
QLoRA
+
small high-quality dataset
```

on your GPU.

Then:

### Project 4

Build:

```text
7B/14B local model
+
LoRA adapter
+
RAG
+
tools
```

That will connect directly to the local coding-agent system you're interested in.

---

# 70. Final "I understand LoRA" test

You should be able to answer these without looking anything up:

**Q1. Why does LoRA exist?**

To adapt a pretrained model while training far fewer parameters than full fine-tuning.

**Q2. What does PEFT mean?**

Parameter-Efficient Fine-Tuning; a family of methods for adapting models while updating relatively few parameters.

**Q3. Is LoRA the same as PEFT?**

No. LoRA is one PEFT method.

**Q4. What is the key LoRA equation?**

$$
W'=W+BA
$$

with scaling often applied.

**Q5. What is frozen?**

The original pretrained model parameters.

**Q6. What is trained?**

The LoRA adapter parameters, primarily $A$ and $B$.

**Q7. What does rank $r$ control?**

The rank/capacity of the low-rank update and therefore the number of trainable adapter parameters.

**Q8. What is SFT?**

A supervised training approach using examples of desired input/output behavior.

**Q9. Can you do SFT with LoRA?**

Yes. That's one of the most common uses.

**Q10. What is QLoRA?**

LoRA applied while the frozen base model is loaded in a quantized representation, commonly 4-bit, to reduce memory requirements.

**Q11. LoRA vs RAG?**

LoRA changes learned model behavior through training; RAG supplies external information at inference time.

**Q12. Why is LoRA useful for your 8-GB GPU?**

It dramatically reduces the number of trainable parameters and associated training memory, while QLoRA can further reduce base-model memory through quantization.

If those 12 answers make intuitive sense rather than just being memorized, **you understand the core of LoRA/PEFT well enough to start implementing it.** The next major step is learning **QLoRA + dataset preparation + an actual Hugging Face training script**, because that's where the theory turns into something you can run on your own machine.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)