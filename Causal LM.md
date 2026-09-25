# Causal LM

I'll explain **Causal Language Modeling (Causal LM) / Next-Token Prediction** from the ground up, then connect it to **Transformers, tokenization, embeddings, QKV attention, training, loss, backpropagation, inference, Hugging Face, and fine-tuning**.

The goal is that after this, you should be able to look at code like `AutoModelForCausalLM`, `labels=input_ids`, or `logits[:, -1, :]` and understand **exactly what is happening**.

---

# 1. What is Causal Language Modeling?

The simplest definition is:

> **A Causal Language Model learns to predict the next token given all the tokens that came before it.**

For example:

```text
The cat sat on the
```

The model should predict:

```text
mat
```

Then:

```text
The cat sat on the mat
```

It predicts the next token, perhaps:

```text
.
```

Then:

```text
The cat sat on the mat.
```

It predicts another token.

This is called **next-token prediction**.

The word **causal** means:

> When predicting token `t`, the model is allowed to look at tokens before `t`, but not tokens after `t`.

That's the central idea.

---

# 2. Why "token" instead of "word"?

LLMs generally don't operate directly on words.

They operate on **tokens**.

For example, a tokenizer might convert:

```text
I love programming
```

into something conceptually like:

```text
["I", " love", " programming"]
```

and then into IDs:

```text
[40, 1842, 9217]
```

The exact IDs depend on the tokenizer.

So when we say:

> "Predict the next word"

we usually technically mean:

> **Predict the next token.**

A token might be:

- an entire word
- part of a word
- punctuation
- whitespace + word
- special token

For example:

```text
unbelievable
```

could potentially become:

```text
["un", "believ", "able"]
```

depending on the tokenizer.

---

# 3. The fundamental mathematical problem

Suppose our token sequence is:

```text
x₁, x₂, x₃, ..., xₙ
```

The model learns:

```text
P(xₜ | x₁, x₂, ..., xₜ₋₁)
```

Read this as:

> Probability of the next token `xₜ`, given all previous tokens.

For example:

```text
P("mat" | "The", "cat", "sat", "on", "the")
```

The model might internally assign probabilities like:

```text
mat       → 0.42
floor     → 0.13
ground    → 0.08
bed       → 0.05
...
```

The model doesn't simply memorize:

```text
"The cat sat on the" → "mat"
```

It learns statistical patterns from enormous amounts of text.

---

# 4. The amazing part: one sentence creates MANY training examples

Suppose we have:

```text
The cat sat on the mat
```

Tokenized:

```text
[The, cat, sat, on, the, mat]
```

We can create training targets:

```text
Input                  Target

The                    cat
The cat                sat
The cat sat            on
The cat sat on         the
The cat sat on the     mat
```

So **one sequence contains many prediction tasks**.

This is extremely important.

The model isn't trained by giving it one sentence and asking:

> "What is the next token?"

Instead, during training, we can give the whole sequence to the Transformer **in one forward pass**.

---

# 5. How the training sequence actually looks

Consider:

```text
I love machine learning
```

Tokens:

```text
I       love       machine       learning
```

We shift the sequence by one position.

### Input

```text
I       love       machine
```

### Target

```text
love    machine    learning
```

Conceptually:

```text
Input token → Expected next token

I           → love
love        → machine
machine     → learning
```

Notice something important:

**The model predicts multiple next tokens simultaneously during training.**

This is one of the most important concepts to understand.

---

# 6. But isn't that cheating?

You might think:

> "Wait. If the model receives `I love machine learning`, how can it learn to predict `love`, `machine`, and `learning` without seeing the future?"

This is where **causal attention** comes in.

Suppose:

```text
I     love     machine     learning
```

At the position of `I`, the model can see:

```text
I
```

but not:

```text
love machine learning
```

At the position of `love`, it can see:

```text
I love
```

but not:

```text
machine learning
```

At `machine`, it can see:

```text
I love machine
```

but not:

```text
learning
```

So the model effectively sees:

```text
Position 1:
I

Position 2:
I love

Position 3:
I love machine

Position 4:
I love machine learning
```

This is enforced using a **causal attention mask**.

---

# 7. The causal mask

This is one of the most important things to understand.

Suppose there are 4 tokens:

```text
I     love     machine     learning
```

The attention mask conceptually looks like:

```text
             Can attend to →

             I   love   machine   learning

I            ✓    ✗       ✗          ✗

love         ✓    ✓       ✗          ✗

machine      ✓    ✓       ✓          ✗

learning     ✓    ✓       ✓          ✓
```

Or mathematically:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

The lower-triangular structure prevents information from the future from leaking backward.

This is why it is called **causal**.

---

# 8. Why can training happen in parallel?

This is one of the coolest aspects of Transformers.

You might imagine training must happen like this:

```text
Predict token 1
↓
Predict token 2
↓
Predict token 3
↓
Predict token 4
```

But that's not necessary during training.

Because the entire target sequence is already known, the model can process:

```text
I love machine learning
```

in parallel.

The causal mask prevents each position from seeing the future.

So the GPU can calculate:

```text
Prediction 1
Prediction 2
Prediction 3
Prediction 4
```

at the same time.

This is called **teacher forcing** in the broader sequence-modeling sense: the correct previous tokens are supplied during training rather than the model's own sampled outputs.

---

# 9. What exactly does the model output?

This is another critical concept.

Suppose the vocabulary contains:

```text
50,000 tokens
```

And our input contains:

```text
5 tokens
```

The model produces something roughly shaped like:

```text
5 × 50,000
```

These are called **logits**.

For every input position, the model produces a score for **every possible next token**.

For example:

```text
Input position 1:

"I"

↓

logits:

love       7.2
am         4.8
like       3.9
have       2.1
...
```

Then:

```text
Input position 2:

"I love"

↓

logits:

machine    8.1
you        3.2
coding     5.4
...
```

---

# 10. Logits → probabilities

The model initially produces **logits**, not probabilities.

Suppose:

```text
mat       = 4.2
floor     = 2.1
banana    = 0.3
```

We convert them into probabilities using **softmax**.

Conceptually:

```text
logits
  ↓
softmax
  ↓
probabilities
```

For example:

```text
mat       → 0.82
floor     → 0.11
banana    → 0.02
...
```

All probabilities sum to:

```text
1
```

---

# 11. What does the model actually learn?

It learns parameters:

```text
θ
```

such that:

```text
P(next token | previous tokens)
```

becomes increasingly accurate.

During training:

```text
text
 ↓
tokenizer
 ↓
token IDs
 ↓
embeddings
 ↓
Transformer
 ↓
logits
 ↓
softmax
 ↓
probabilities
 ↓
loss
 ↓
backpropagation
 ↓
update weights
```

This cycle happens **billions/trillions of times across training data**, depending on the model and training setup.

---

# 12. The loss function

Now suppose the correct next token is:

```text
mat
```

The model predicts:

```text
mat       → 0.10
floor     → 0.30
banana    → 0.01
...
```

That's bad because the correct answer has low probability.

We calculate a loss, typically **cross-entropy loss**.

For the correct class, conceptually:

```text
Loss = -log(P(correct token))
```

So:

If:

```text
P(correct) = 0.9
```

then:

```text
Loss ≈ 0.105
```

Good.

If:

```text
P(correct) = 0.01
```

then:

```text
Loss ≈ 4.605
```

Bad.

The model is therefore encouraged to assign higher probability to the correct next token.

---

# 13. A complete miniature example

Suppose:

```text
The dog eats food
```

Tokens:

```text
The
dog
eats
food
```

Training pairs:

```text
The              → dog
The dog          → eats
The dog eats     → food
```

Now imagine the model predicts:

### Position 1

```text
dog      0.70
cat      0.20
car      0.10
```

Correct:

```text
dog
```

Good.

### Position 2

```text
eats     0.60
runs     0.30
sleeps   0.10
```

Correct:

```text
eats
```

Good.

### Position 3

```text
food     0.20
grass    0.50
meat     0.30
```

Correct:

```text
food
```

Bad.

The loss combines these errors.

The optimizer then modifies the neural network's parameters so that, ideally, future predictions become better.

---

# 14. Where does the Transformer come into this?

The causal LM objective doesn't itself specify the architecture.

Historically, language models have used different architectures.

But modern GPT-style causal LMs generally use a **decoder-only Transformer**.

The flow is approximately:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Token embeddings + positional information
 ↓
Transformer blocks
 ↓
Hidden representations
 ↓
LM head
 ↓
Logits
 ↓
Probability distribution
```

You have already been learning the pieces:

```text
Tokenization
      ↓
Embeddings
      ↓
Attention Q/K/V
      ↓
Transformer architecture
      ↓
Causal LM
```

These aren't separate unrelated topics.

They fit together into one pipeline.

---

# 15. Let's connect Q/K/V to next-token prediction

Suppose:

```text
The cat sat on the
```

The model needs to determine what information is useful for predicting the next token.

Attention allows different tokens to interact.

For example, when processing:

```text
The cat sat on the
```

the representation at the final position might pay attention strongly to:

```text
cat
sat
on
```

and perhaps other relevant context.

The attention mechanism computes:

```text
Q = Query
K = Key
V = Value
```

and approximately:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √dₖ)V
```

with a causal mask applied so future positions cannot be accessed.

The result is a contextual representation.

Then the Transformer layers repeatedly refine those representations.

Finally:

```text
hidden state
      ↓
linear layer / LM head
      ↓
vocabulary logits
```

---

# 16. What is the LM head?

Suppose the final hidden representation is:

```text
h
```

The model needs to convert it into scores for every vocabulary token.

A linear layer does approximately:

```text
logits = hW + b
```

If vocabulary size is:

```text
50,000
```

then the output has approximately:

```text
50,000 scores
```

One score per possible next token.

So:

```text
Transformer
    ↓
hidden state
    ↓
LM head
    ↓
50,000 logits
```

This is why you'll see models called:

```python
AutoModelForCausalLM
```

in Hugging Face.

It means:

> Give me a Transformer model equipped with a language-modeling head for causal next-token prediction.

---

# 17. The most important training trick: shifting labels

This is something you will encounter constantly in code.

Suppose:

```python
input_ids = [10, 20, 30, 40]
```

Conceptually:

```text
10 → 20
20 → 30
30 → 40
```

Therefore:

```text
inputs:

10 20 30 40

targets:

20 30 40 ?
```

The exact implementation depends on the model/framework, but Hugging Face causal-LM models commonly let you simply provide:

```python
labels=input_ids
```

and the model internally handles the appropriate shift when calculating the causal language-modeling loss.

This is extremely convenient.

---

# 18. Why `labels=input_ids` works

Suppose:

```python
input_ids = [
    10,
    20,
    30,
    40
]
```

The model effectively evaluates:

```text
10 → 20
20 → 30
30 → 40
```

So you don't necessarily have to manually construct:

```python
labels = [20, 30, 40, ...]
```

The model's loss implementation performs the shifting.

You'll see code like:

```python
outputs = model(
    input_ids=input_ids,
    labels=input_ids
)
```

and:

```python
loss = outputs.loss
```

This is a standard causal-LM training pattern.

---

# 19. Training vs inference

This distinction is **extremely important**.

## Training

You have:

```text
The cat sat on the mat
```

The model can process the entire sequence in parallel using causal masking.

It learns:

```text
The       → cat
The cat   → sat
The cat sat → on
...
```

---

## Inference

Now you give:

```text
The cat sat on the
```

The model predicts:

```text
mat
```

Then you append it:

```text
The cat sat on the mat
```

Then run again:

```text
The cat sat on the mat
```

and predict:

```text
.
```

Then:

```text
The cat sat on the mat.
```

and predict the next token.

So generation is **autoregressive**.

---

# 20. Autoregressive means what?

It means:

> The model uses its previously generated output as part of the input for generating the next output.

Like:

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
predict next token
 ↓
...
```

For example:

```text
User:
What is Docker?

Model:
Docker

↓

Docker is

↓

Docker is a

↓

Docker is a platform

↓

Docker is a platform for

↓

Docker is a platform for building
```

Each generated token becomes part of the context for subsequent predictions.

---

# 21. Why does ChatGPT generate one token at a time?

Because the next token depends on the previous generated tokens.

Suppose:

```text
The capital of France is
```

The model predicts:

```text
Paris
```

Now it has:

```text
The capital of France is Paris
```

The next token might be:

```text
.
```

The model can't know the exact sequence it will generate before generating it because generation depends on the previous outputs.

Therefore inference is sequential.

---

# 22. Training is parallel, generation is sequential

Remember this:

### Training

```text
parallel
```

### Generation

```text
autoregressive / sequential
```

This difference is fundamental to understanding LLM performance.

---

# 23. Why are LLMs so expensive to run?

Suppose the model has:

```text
7 billion parameters
```

For every generated token, the model has to perform a huge amount of computation.

And generation might require:

```text
token 1
↓
token 2
↓
token 3
↓
...
token 1000
```

That's why **tokens/second** is such an important metric for local LLMs.

---

# 24. What is temperature?

Once you have probabilities:

```text
Paris       0.70
London      0.15
Berlin      0.08
Madrid      0.04
...
```

you need a strategy for choosing the next token.

One option:

```text
always choose highest probability
```

This is roughly **greedy decoding**.

Temperature changes the sharpness of the probability distribution.

Conceptually:

```text
low temperature
→ more deterministic

high temperature
→ more varied/random
```

For example:

```text
temperature = 0
```

approximately pushes generation toward deterministic behavior, though exact implementation details vary.

Higher temperature makes lower-probability tokens more likely to be sampled.

---

# 25. Top-k and top-p

Generation doesn't necessarily sample from every token in the vocabulary.

### Top-k

Keep only the:

```text
k
```

highest-probability tokens.

For example:

```text
top_k = 50
```

means:

> Consider the 50 highest-probability tokens.

---

### Top-p

Also called **nucleus sampling**.

Instead of selecting a fixed number of tokens, choose the smallest set whose cumulative probability reaches:

```text
p
```

For example:

```text
top_p = 0.9
```

means approximately:

> Consider the smallest group of likely tokens whose combined probability reaches 90%.

---

# 26. What does "the model knows" actually mean?

This is an important conceptual point.

During pretraining, the model doesn't receive a database like:

```text
France → Paris
India → New Delhi
```

Instead, it adjusts billions of parameters based on prediction errors.

Through training, its parameters encode statistical patterns involving:

- words
- syntax
- facts
- concepts
- relationships
- programming patterns
- reasoning patterns
- styles
- languages
- structures

So the fundamental training task remains surprisingly simple:

> **Predict the next token.**

Yet a sufficiently large model trained on sufficiently broad data can acquire many capabilities.

---

# 27. Why does next-token prediction produce intelligence-like behavior?

This is one of the deepest questions in LLMs.

Consider:

```text
The woman went to the hospital because she was
```

To predict the next token accurately, the model needs to understand relationships.

Possible continuation:

```text
sick
```

Now consider:

```text
The woman went to the hospital because she was carrying a
```

Maybe:

```text
baby
```

To make good predictions, the model needs to capture patterns involving:

```text
grammar
context
semantics
world knowledge
relationships
style
```

So although the training objective is:

```text
predict next token
```

solving that objective well requires learning increasingly sophisticated representations.

---

# 28. An important misconception

You may hear:

> "An LLM is just predicting the next word."

That's technically true but often misleading.

It is like saying:

> "A chess engine is just predicting the next move."

Technically yes.

But predicting good moves requires a huge internal model of the game.

Similarly, predicting tokens well requires modeling many properties of language and the data that language describes.

---

# 29. Causal LM vs masked language model

You will encounter this distinction when working with Hugging Face.

### Causal LM

Predict:

```text
future token
```

using previous tokens.

Example:

```text
The cat sat on the ___
```

Model predicts:

```text
mat
```

Typical architecture:

```text
decoder-only Transformer
```

Examples include many GPT-style models.

---

### Masked Language Model

Instead, hide a token:

```text
The cat sat on the [MASK]
```

and ask the model to recover it.

A masked model can generally use information from **both sides** of the masked position.

Typical architecture:

```text
encoder-only Transformer
```

Classic example:

```text
BERT
```

So:

```text
Causal LM:
left → right

Masked LM:
both directions around masked positions
```

---

# 30. Why is Causal LM particularly useful for chatbots?

Because conversation naturally looks like:

```text
User:
Explain Docker.

Assistant:
Docker is...
```

The model can continue the sequence.

The entire conversation can be represented as tokens:

```text
User tokens
+
Assistant tokens
```

The model predicts the assistant's next tokens.

Modern chat models add special formatting/control tokens to distinguish roles and structure conversations, but the fundamental generation mechanism remains causal next-token prediction.

---

# 31. Chat fine-tuning

Suppose we have:

```text
User:
What is recursion?

Assistant:
Recursion is a technique...
```

During supervised fine-tuning, examples like this can teach the model to generate appropriate responses.

The objective remains largely:

```text
predict next token
```

but the training data now emphasizes desired conversational behavior.

So:

```text
Pretraining
↓
general language patterns

Instruction/chat fine-tuning
↓
follow instructions / conversational behavior
```

---

# 32. Pretraining vs fine-tuning

This distinction will matter enormously when you start LoRA.

### Pretraining

Train a model on enormous amounts of text:

```text
books
web pages
code
articles
documents
etc.
```

Objective:

```text
next-token prediction
```

Result:

```text
general-purpose language model
```

---

### Fine-tuning

Take an existing model:

```text
7B model
```

and train it further on a specialized dataset.

For example:

```text
Python coding dataset
```

or:

```text
medical Q&A
```

or:

```text
instruction-following conversations
```

The basic next-token loss can remain the training objective.

---

# 33. LoRA and causal LM

This directly connects to your interest in fine-tuning.

Suppose you have:

```text
Qwen / Llama / Mistral-type causal LM
```

You don't necessarily want to update all billions of parameters.

LoRA adds trainable low-rank adapters to selected layers.

Conceptually:

```text
Original model
       +
LoRA adapters
       ↓
fine-tuned behavior
```

The model still performs:

```text
input
 ↓
next-token logits
 ↓
loss
 ↓
backpropagation
```

but only selected parameters/adapters are updated.

---

# 34. A practical Hugging Face example

A simplified example looks like:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "..."

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text = "The capital of France is"

inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

logits = outputs.logits
```

Now:

```python
logits.shape
```

will conceptually be:

```text
(batch_size, sequence_length, vocabulary_size)
```

For example:

```text
(1, 6, 50000)
```

means:

```text
1 sequence
6 input positions
50000 possible tokens
```

---

# 35. Getting the next-token prediction

Suppose:

```python
logits = outputs.logits
```

We only want the prediction corresponding to the **last input position**:

```python
next_token_logits = logits[:, -1, :]
```

Why?

Because:

```text
logits[:, -1, :]
```

contains the scores for:

> What token should come after the current sequence?

Then:

```python
next_token_id = next_token_logits.argmax(dim=-1)
```

selects the highest-scoring token.

Then:

```python
tokenizer.decode(next_token_id)
```

converts the token ID back into text.

Conceptually:

```text
"The capital of France is"
              ↓
          Transformer
              ↓
       vocabulary logits
              ↓
          argmax
              ↓
           Paris
```

---

# 36. But real generation is more complicated

Instead of manually doing:

```python
argmax
```

you'll commonly use:

```python
model.generate(...)
```

For example:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100
)
```

The generation system handles repeated prediction:

```text
predict
↓
append
↓
predict
↓
append
↓
...
```

and can apply:

- temperature
- top-k
- top-p
- repetition penalties
- stopping criteria
- sampling
- caching

---

# 37. KV cache

This is particularly important when you start understanding LLM inference performance.

Suppose you've already processed:

```text
The cat sat on the
```

Then you generate:

```text
mat
```

You don't want to completely recompute all attention information from scratch every time.

Transformers can cache previously computed **keys and values**.

That's the:

> **KV cache**

You learned Q/K/V earlier.

During autoregressive generation:

```text
previous K,V
+
new token's K,V
```

can be reused.

This substantially reduces redundant computation during generation.

---

# 38. Context window

Causal LM doesn't mean infinite memory.

The model can only process a finite number of tokens in its context window.

For example, a hypothetical model might support:

```text
128k tokens
```

Then:

```text
conversation
documents
instructions
code
```

must fit within that context limit, subject to the model/system's actual limits.

The context window is different from the model's parameter count.

For example:

```text
7B parameters
128k context
```

are two completely different properties.

---

# 39. What happens if the context is too long?

Depending on the system, older content may need to be:

- truncated
- summarized
- retrieved again from external storage
- otherwise managed by the application

This is where **RAG** becomes relevant.

RAG can retrieve relevant information and put it into the model's current context.

But the actual model still performs:

```text
next-token prediction
```

---

# 40. Causal LM + RAG

Suppose you have:

```text
50 PDFs
```

You create:

```text
PDFs
 ↓
chunks
 ↓
embeddings
 ↓
vector database
```

Then user asks:

```text
What does chapter 4 say about Docker networking?
```

RAG retrieves relevant chunks:

```text
relevant document chunks
```

Then your application constructs:

```text
System instructions
+
retrieved context
+
user question
```

and sends them to the causal LM.

The model then generates:

```text
answer token 1
↓
answer token 2
↓
answer token 3
...
```

Again:

> RAG supplies information; causal LM generates the answer.

---

# 41. Causal LM and hallucinations

Next-token prediction does **not** inherently mean:

> "Only say things you know are true."

The model's objective is to generate likely continuations.

Suppose the prompt strongly suggests that something exists even though it doesn't.

The model may generate a plausible continuation.

That's one reason LLMs can hallucinate.

RAG, tool use, verification, better training, and system design can reduce this problem, but next-token prediction itself is not a truth-verification mechanism.

---

# 42. The complete picture

You can now connect almost everything you've been studying:

```text
                    RAW TEXT
                       │
                       ▼
                 TOKENIZATION
                       │
                       ▼
                  TOKEN IDs
                       │
                       ▼
                   EMBEDDINGS
                       │
                       ▼
             POSITION INFORMATION
                       │
                       ▼
              TRANSFORMER BLOCK
                       │
          ┌────────────┴────────────┐
          │                         │
      Q / K / V                Feed Forward
          │                         │
          └────────────┬────────────┘
                       ▼
                 Hidden states
                       │
                       ▼
                    LM HEAD
                       │
                       ▼
                    LOGITS
                       │
                       ▼
                   SOFTMAX
                       │
                       ▼
                TOKEN PROBABILITIES
                       │
                       ▼
              COMPARE WITH TARGET
                       │
                       ▼
                    LOSS
                       │
                       ▼
               BACKPROPAGATION
                       │
                       ▼
                WEIGHT UPDATE
```

That's **training**.

---

# 43. Generation

During generation:

```text
Prompt
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Transformer
  ↓
Logits
  ↓
Sampling / decoding
  ↓
Next token
  ↓
Append token
  ↓
Transformer again
  ↓
Next token
  ↓
...
```

Eventually:

```text
tokens
 ↓
detokenizer
 ↓
text
```

---

# 44. One example from beginning to end

Let's use:

```text
The dog is
```

Suppose tokenization gives:

```text
[The, dog, is]
```

### Step 1 — Embedding

Each token becomes a vector:

```text
The → [0.12, -0.31, ...]
dog → [0.71,  0.18, ...]
is  → [-0.22, 0.91, ...]
```

These vectors aren't just dictionary definitions; they are learned representations.

---

### Step 2 — Transformer

The model processes these vectors.

Attention allows information to flow between positions while causal masking prevents future information leakage.

---

### Step 3 — Final representation

At the last position, the model obtains a contextual hidden vector representing the current context:

```text
"The dog is ..."
```

---

### Step 4 — LM head

The hidden vector is converted into vocabulary logits:

```text
sleeping → 8.2
running  → 6.4
eating   → 7.1
...
```

---

### Step 5 — Softmax

Maybe:

```text
sleeping → 0.55
eating   → 0.20
running  → 0.15
...
```

---

### Step 6 — Decode

The generation algorithm chooses:

```text
sleeping
```

Now the sequence becomes:

```text
The dog is sleeping
```

---

### Step 7

The model predicts another token:

```text
.
```

Result:

```text
The dog is sleeping.
```

---

# 45. What does the model learn at each layer?

Don't think of this as a strict universal rule, because modern Transformer representations are distributed and layer behavior varies.

But conceptually, different layers can develop increasingly sophisticated representations involving:

```text
tokens
↓
local patterns
↓
syntax
↓
relationships
↓
semantic information
↓
long-range dependencies
↓
higher-level abstractions
```

The exact behavior isn't as simple as:

> Layer 1 = grammar, Layer 2 = facts, Layer 3 = reasoning.

Real models are more distributed than that.

---

# 46. Why more parameters can help

A larger model has more parameters available to represent complex patterns.

For example:

```text
1B
7B
14B
32B
70B
```

The parameter count refers roughly to the number of learned numerical parameters.

It does **not** directly mean:

```text
70B = 10× smarter than 7B
```

There isn't such a simple relationship.

Performance depends on:

- architecture
- training data
- data quality
- training compute
- parameter count
- context length
- optimization
- fine-tuning
- inference setup

---

# 47. Why data quality matters

Suppose training data contains:

```text
high-quality explanations
```

The model learns patterns from those examples.

If the training data contains huge quantities of:

```text
incorrect information
```

the model can also learn those patterns.

Next-token prediction doesn't magically determine whether training text is true.

It learns statistical regularities in the training distribution.

---

# 48. What is perplexity?

You will encounter **perplexity** when studying language-model evaluation.

Very roughly:

> Perplexity measures how surprised the model is by the actual next tokens.

Lower perplexity generally means the model assigns higher probability to the observed text.

Mathematically, if average negative log-likelihood is:

```text
L
```

then:

```text
Perplexity = exp(L)
```

Lower is generally better **when comparing models under the same evaluation setup and dataset**.

But perplexity isn't a complete measure of chatbot quality.

---

# 49. Cross-entropy and perplexity relationship

Suppose:

```text
average loss = 2
```

Then:

```text
perplexity = e²
           ≈ 7.39
```

So:

```text
loss ↓
perplexity ↓
```

generally indicates better predictive performance on that evaluation corpus.

---

# 50. Teacher forcing

This is worth understanding clearly.

During training, suppose:

```text
The cat sat on the mat
```

The model receives the actual previous tokens:

```text
The
The cat
The cat sat
The cat sat on
...
```

It does **not** have to use its own previous prediction.

That is teacher forcing.

During generation, however:

```text
model prediction
      ↓
becomes input
      ↓
next prediction
```

This difference creates a distinction between training and inference.

---

# 51. Exposure bias

Because training uses correct previous tokens while inference uses the model's own generated tokens, errors can compound during generation.

For example:

```text
correct → correct → wrong → strange → stranger → ...
```

Modern LLM training and decoding strategies have various ways of dealing with this broader issue, but the underlying distinction is important.

---

# 52. Causal attention vs causal LM

Don't confuse these two.

### Causal attention

A mechanism that prevents a token from attending to future tokens.

```text
future information blocked
```

### Causal language modeling

The training objective:

```text
predict next token
```

Usually a GPT-style model uses **causal attention** to implement **causal language modeling**.

So:

```text
Causal attention
      ↓
allows proper autoregressive information flow
      ↓
Causal LM objective
      ↓
next-token prediction
```

---

# 53. The single most important equation

If you remember only one mathematical expression, remember:

$$
P(x_t \mid x_1,x_2,\ldots,x_{t-1})
$$

It means:

> Probability of token `xₜ` given all previous tokens.

The model tries to maximize:

$$
P(x_1,x_2,\ldots,x_n)
$$

which can be factorized as:

$$
P(x_1,x_2,\ldots,x_n)
=
\prod_{t=1}^{n}
P(x_t\mid x_1,\ldots,x_{t-1})
$$

This is the mathematical foundation of autoregressive language modeling.

---

# 54. The loss equation

Training usually minimizes the negative log-likelihood:

$$
L
=
-\sum_{t=1}^{n}
\log P(x_t\mid x_{<t})
$$

or its average over tokens.

This says:

> Penalize the model when it gives low probability to the actual next token.

That's essentially what the training process is trying to optimize.

---

# 55. A very useful mental model

Think of the model as a gigantic function:

$$
f(\text{previous tokens}) \rightarrow \text{probability distribution over next tokens}
$$

For example:

```text
"I am going to the"
          ↓
      Transformer
          ↓
┌──────────────────────┐
│ beach      0.32       │
│ store      0.17       │
│ market     0.12       │
│ gym        0.08       │
│ ...                   │
└──────────────────────┘
```

The model isn't directly outputting:

```text
"beach"
```

It outputs a distribution.

The decoding algorithm chooses a token from that distribution.

---

# 56. What happens inside `generate()`?

Conceptually:

```python
while not finished:

    outputs = model(input_ids)

    logits = outputs.logits[:, -1, :]

    next_token = choose_token(logits)

    input_ids = concatenate(
        input_ids,
        next_token
    )
```

Then repeat.

Actual implementations optimize this heavily using things like:

```text
KV cache
batching
GPU kernels
sampling implementations
attention optimizations
```

but conceptually this is what happens.

---

# 57. Practical experiment you should actually do

Since you're learning this for practical LLM work, don't just memorize it.

Run a tiny experiment.

Install:

```bash
pip install torch transformers
```

Then:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text = "The capital of France is"

inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

print("Input IDs:")
print(inputs["input_ids"])

print("\nLogits shape:")
print(outputs.logits.shape)

next_token_logits = outputs.logits[:, -1, :]

next_token_id = next_token_logits.argmax(dim=-1)

print("\nPredicted token:")
print(tokenizer.decode(next_token_id))
```

Pay particular attention to:

```python
outputs.logits.shape
```

and:

```python
outputs.logits[:, -1, :]
```

Once you understand those two things, the architecture becomes much less mysterious.

---

# 58. Then try actual generation

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50
)

print(tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
))
```

You should now be able to understand that this isn't magic.

It is essentially:

```text
prompt
 ↓
tokenize
 ↓
forward pass
 ↓
next-token distribution
 ↓
choose token
 ↓
append
 ↓
forward pass
 ↓
choose token
 ↓
...
```

---

# 59. Then inspect probabilities

You can go one step deeper:

```python
import torch

logits = outputs.logits[:, -1, :]

probs = torch.softmax(logits, dim=-1)

top_probs, top_ids = torch.topk(probs, 10)

for probability, token_id in zip(top_probs[0], top_ids[0]):
    token = tokenizer.decode([token_id])
    print(token, probability.item())
```

You will literally see something like:

```text
Paris      0.XX
London     0.XX
...
```

The exact results depend on the model and prompt.

This is an excellent experiment because you can **see next-token prediction happening**.

---

# 60. The complete LLM stack you should now understand

You are currently studying these topics in roughly this order:

```text
1. Tokenization
        ↓
2. Embeddings
        ↓
3. Q/K/V Attention
        ↓
4. Transformer Architecture
        ↓
5. Causal LM / Next-token prediction
        ↓
6. Pretraining
        ↓
7. Fine-tuning
        ↓
8. LoRA / PEFT
        ↓
9. Quantization
        ↓
10. Inference
        ↓
11. RAG
        ↓
12. Agents / tool use
```

And the connections are:

```text
Tokenization
    ↓
tokens

Embeddings
    ↓
vectors

Q/K/V
    ↓
tokens interact

Transformer
    ↓
contextual representations

Causal LM
    ↓
predict next token

Loss
    ↓
measure prediction error

Backpropagation
    ↓
change weights

Fine-tuning
    ↓
adapt model

LoRA
    ↓
efficient adaptation

Quantization
    ↓
reduce memory/computation

RAG
    ↓
provide external information

Agent
    ↓
use model + tools
```

---

# 61. The 10 things I want you to remember

If you understand these, you understand causal LM:

### 1.

**A causal LM predicts the next token.**

```text
A B C → D
```

### 2.

It models:

$$
P(x_t|x_{<t})
$$

### 3.

**Causal attention prevents looking into the future.**

```text
past ✓
future ✗
```

### 4.

During training, many next-token predictions happen **in parallel**.

### 5.

The model outputs:

```text
logits
```

for every vocabulary token.

### 6.

Softmax converts logits into probabilities.

### 7.

Cross-entropy measures how much probability the model assigned to the correct token.

### 8.

Backpropagation updates the model's parameters.

### 9.

During generation, the model works autoregressively:

```text
predict → append → predict → append → ...
```

### 10.

**Modern GPT-style LLMs are fundamentally trained to predict tokens, but doing that extremely well requires learning rich representations of language and the patterns present in their training data.**

---

# 62. The one diagram to memorize

If you can explain this diagram yourself, you have understood causal LM:

```text
                 TRAINING

              "The cat sat"
                    │
                    ▼
               TOKENIZER
                    │
                    ▼
             [The, cat, sat]
                    │
                    ▼
               EMBEDDINGS
                    │
                    ▼
           ┌─────────────────┐
           │   TRANSFORMER   │
           │                 │
           │  Causal Q/K/V   │
           │       +         │
           │  Feed Forward   │
           └────────┬────────┘
                    │
                    ▼
                  LOGITS
                    │
                    ▼
                SOFTMAX
                    │
                    ▼
       Probability of next token
                    │
                    ▼
            Compare with target
                    │
                    ▼
              CROSS-ENTROPY
                    │
                    ▼
             BACKPROPAGATION
                    │
                    ▼
              UPDATE WEIGHTS
```

And during inference:

```text
             "The cat"
                 │
                 ▼
             TRANSFORMER
                 │
                 ▼
             NEXT TOKEN
                 │
                "sat"
                 │
                 ▼
       "The cat sat"
                 │
                 ▼
             TRANSFORMER
                 │
                 ▼
             NEXT TOKEN
                 │
                "on"
                 │
                 ▼
              repeat...
```

**That is the fundamental engine behind a decoder-only LLM.**

And now, when you move to **pretraining, fine-tuning and LoRA**, the key question becomes much simpler: *what data do we give this next-token predictor, what tokens do we calculate the loss on, and which parameters do we allow to change?*
