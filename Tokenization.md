# Tokenization

I'll explain tokenization from the ground up and connect it directly to what actually happens inside an LLM.

# Tokenization — From Text to Numbers

The single most important idea is:

> **An LLM does not read words, sentences, or characters directly. It reads sequences of numbers called token IDs.**

Tokenization is the process that converts:

**human text → tokens → token IDs → embeddings → Transformer**

For example:

```text
"I love machine learning!"
          ↓
["I", " love", " machine", " learning", "!"]
          ↓
[40, 3021, 4987, 5219, 0]
          ↓
vectors
          ↓
Transformer
```

The exact tokens and IDs depend on the tokenizer/model.

---

# 1. Why do we need tokenization?

Suppose we give an LLM:

```text
The cat is sleeping.
```

A computer doesn't inherently understand:

```text
The
cat
is
sleeping
```

as concepts.

A neural network ultimately operates on **numbers**.

So we need a pipeline:

```text
Text
 ↓
Tokenization
 ↓
Token IDs
 ↓
Embedding lookup
 ↓
Vectors
 ↓
Transformer
```

For example:

```text
"The cat is sleeping"
```

might become:

```text
["The", " cat", " is", " sleeping"]
```

then:

```text
[791, 8415, 374, 2847]
```

and then each number gets converted into a vector:

```text
791  → [0.12, -0.43, 0.81, ...]
8415 → [0.31,  0.22, -0.19, ...]
374  → [-0.51, 0.72, 0.08, ...]
...
```

Those vectors are what enter the Transformer.

---

# 2. What exactly is a token?

A **token is a piece of text chosen by a tokenizer**.

This is extremely important:

> A token is **not necessarily a word**.

For example:

```text
I love cats
```

could become:

```text
["I", " love", " cats"]
```

But:

```text
unhappiness
```

could become:

```text
["un", "happi", "ness"]
```

Or something like:

```text
["un", "happiness"]
```

depending on the tokenizer.

And:

```text
ChatGPT
```

could potentially become:

```text
["Chat", "GPT"]
```

or:

```text
["ChatGPT"]
```

or several smaller pieces.

The tokenizer determines this.

---

# 3. Tokens can be words

Older/simple tokenizers can use **word-level tokenization**.

Example:

```text
I love machine learning
```

becomes:

```text
["I", "love", "machine", "learning"]
```

Vocabulary:

```text
I          → 0
love       → 1
machine    → 2
learning   → 3
```

Then:

```text
"I love machine learning"
```

becomes:

```text
[0, 1, 2, 3]
```

Seems perfect.

But there's a huge problem.

---

# 4. The vocabulary problem

Imagine trying to create a token for **every possible word**.

English has hundreds of thousands of words.

But consider:

```text
play
plays
played
playing
playful
playfulness
player
players
```

Then names:

```text
Aditya
Priyanshu
OpenAI
HuggingFace
```

Then technical terms:

```text
transformer
tokenization
backpropagation
quantization
```

Then typos:

```text
transfomer
tokeniztion
```

Then completely new words.

You can't realistically have a vocabulary containing every possible word.

And this gets dramatically worse with:

- programming languages
- URLs
- usernames
- mathematical notation
- emojis
- multiple languages
- names
- scientific terminology

So modern LLMs generally don't use pure word-level tokenization.

---

# 5. Character-level tokenization

Another possibility is:

```text
"hello"
```

→

```text
["h", "e", "l", "l", "o"]
```

Vocabulary could be tiny:

```text
a
b
c
...
z
0
1
2
...
```

This solves the vocabulary problem.

But now:

```text
"Transformers are powerful"
```

might require dozens of tokens.

That's inefficient.

The model has to process much more sequence length.

And sequence length matters enormously for Transformers.

So we need something between:

**whole words**

and

**individual characters**.

That's where **subword tokenization** comes in.

---

# 6. Subword tokenization

Modern LLMs generally tokenize text into **subword pieces**.

Instead of:

```text
playing
```

being either:

```text
["playing"]
```

or:

```text
["p", "l", "a", "y", "i", "n", "g"]
```

it might become:

```text
["play", "ing"]
```

This gives us the best of both worlds.

Common pieces can be stored as tokens:

```text
the
ing
tion
ing
machine
learn
```

while uncommon words can be broken apart.

For example:

```text
unbelievable
```

might become:

```text
["un", "believ", "able"]
```

The exact result depends on the tokenizer.

---

# 7. The important concept: vocabulary

A tokenizer has a **vocabulary**.

Think of it as a dictionary:

```text
token → ID
```

For example, a toy tokenizer might have:

| Token | ID |
|---|---:|
| `<pad>` | 0 |
| `<unk>` | 1 |
| `I` | 2 |
| ` love` | 3 |
| ` machine` | 4 |
| ` learning` | 5 |
| `!` | 6 |

Then:

```text
"I love machine learning!"
```

could become:

```text
["I", " love", " machine", " learning", "!"]
```

and then:

```text
[2, 3, 4, 5, 6]
```

These numbers are called **token IDs**.

---

# 8. Token ID does NOT mean meaning

This is a very important misconception.

Suppose:

```text
cat → 500
dog → 501
```

It does **not** mean:

```text
501 is mathematically close to 500
```

or that:

```text
dog = cat + 1
```

The IDs are simply identifiers.

Think of them like database IDs:

```text
Aditya → student ID 10452
Rahul  → student ID 10453
```

The number itself has no semantic meaning.

Meaning starts being represented when token IDs are converted into **embeddings**.

---

# 9. Tokenization pipeline

Now let's connect everything.

Suppose you write:

```text
I love Transformers.
```

### Step 1 — Raw text

```text
"I love Transformers."
```

### Step 2 — Tokenizer splits it

Potentially:

```text
["I", " love", " Transformers", "."]
```

### Step 3 — Tokens become IDs

```text
[40, 3021, 14892, 13]
```

### Step 4 — IDs become embeddings

Each ID indexes a row in the model's embedding matrix.

Conceptually:

```text
40     → [0.2, -0.7, 0.1, ...]
3021   → [0.5,  0.3, 0.8, ...]
14892  → [-0.1, 0.9, 0.2, ...]
13     → [0.4, -0.2, 0.7, ...]
```

### Step 5 — Transformer processes them

```text
Embeddings
     ↓
Positional information
     ↓
Self-Attention
     ↓
Feed Forward Network
     ↓
...
     ↓
Output probabilities
```

This is why understanding tokenization is essential before understanding Transformers deeply.

---

# 10. Why do you sometimes see spaces inside tokens?

This confuses almost everyone initially.

You might see:

```text
["Hello", " world", "!"]
```

instead of:

```text
["Hello", "world", "!"]
```

Why?

Because some tokenizers encode the **space as part of the following token**.

So:

```text
"Hello world"
```

might become:

```text
["Hello", " world"]
```

The second token represents:

```text
" world"
```

including the leading space.

This allows the tokenizer to distinguish things like:

```text
hello
 hello
```

depending on the tokenizer scheme.

---

# 11. Different tokenizers tokenize differently

This is extremely important.

There is **no universal tokenization**.

For example, the same sentence:

```text
I love artificial intelligence.
```

might be tokenized differently by different models.

Model A:

```text
["I", " love", " artificial", " intelligence", "."]
```

Model B:

```text
["I", " love", " art", "ificial", " intelligence", "."]
```

Model C:

```text
["I", " love", " artificial", " intellig", "ence", "."]
```

Therefore:

> **Token IDs only make sense relative to their tokenizer/model.**

Token ID `5000` in one model has nothing to do with token ID `5000` in another model.

---

# 12. The major tokenization algorithms

You should know these names because you'll encounter them constantly in NLP/LLM material.

## 12.1 BPE — Byte Pair Encoding

**BPE** is one of the most important tokenization algorithms.

The basic idea is:

> Start with small units and repeatedly merge frequently occurring pairs.

Imagine our training text contains:

```text
low
lower
lowest
```

Initially:

```text
l o w
l o w e r
l o w e s t
```

If:

```text
l + o
```

appears frequently:

```text
l o → lo
```

Then:

```text
lo + w → low
```

Eventually the vocabulary can contain:

```text
low
lower
est
er
```

etc.

The tokenizer learns useful pieces from the training corpus.

---

# 13. Why BPE works

Suppose the vocabulary contains:

```text
the
ing
tion
machine
learn
```

Then:

```text
machinelearning
```

doesn't necessarily need to be completely unknown.

It can be decomposed:

```text
machine + learn + ing
```

So the tokenizer can handle words it never explicitly encountered as a complete word.

This is one of the major reasons subword tokenization works so well.

---

# 14. WordPiece

WordPiece is another subword algorithm.

It is associated particularly with models such as BERT.

Instead of simply asking:

> "Which pair occurs most frequently?"

the algorithm considers which subword vocabulary provides a useful representation of the training data.

You don't need to memorize the mathematical details initially.

Understand the purpose:

```text
Words
 ↓
Reusable subword pieces
 ↓
Vocabulary
```

---

# 15. Unigram tokenization

Another approach is the **Unigram** model.

Instead of building the vocabulary primarily through repeated merging, it starts with a relatively large collection of candidate pieces and removes pieces that aren't useful.

You can think of it as:

```text
Large candidate vocabulary
        ↓
Evaluate pieces
        ↓
Remove less useful pieces
        ↓
Final vocabulary
```

SentencePiece commonly supports Unigram tokenization.

---

# 16. SentencePiece

You will encounter **SentencePiece** frequently.

Important:

> SentencePiece is a tokenizer framework/library, not simply one single tokenization algorithm.

It can implement approaches such as:

- BPE
- Unigram

One advantage is that it can operate directly on raw text without requiring traditional whitespace-based word splitting first.

This is particularly useful for multilingual models.

---

# 17. Byte-level tokenization

Now we get to an important modern LLM concept.

Some tokenizers operate at the **byte level**.

Why?

Because bytes provide a way of representing essentially arbitrary text.

For example:

```text
English
Hindi
中文
العربية
emoji 😀
```

can ultimately be represented as bytes.

This helps avoid the traditional `<UNK>` problem where an unknown character/word cannot be represented.

Models such as GPT-style systems have used byte-level BPE variants.

---

# 18. What is `<UNK>`?

`<UNK>` means:

```text
unknown token
```

Imagine a vocabulary containing:

```text
cat
dog
house
```

and you encounter:

```text
xyzabc
```

A basic word tokenizer might say:

```text
<UNK>
```

Modern subword/byte-based tokenizers often greatly reduce or eliminate the need for `<UNK>` because they can break unfamiliar text into smaller pieces.

---

# 19. Special tokens

Tokenizers also have special tokens that aren't ordinary language.

Examples include:

```text
<BOS>
<EOS>
<PAD>
<UNK>
<CLS>
<SEP>
```

Their exact names vary between models.

### BOS

Beginning of sequence:

```text
<BOS>
```

### EOS

End of sequence:

```text
<EOS>
```

### PAD

Padding:

```text
<PAD>
```

### UNK

Unknown token:

```text
<UNK>
```

Some modern LLMs use different names or don't use all of these.

---

# 20. Why do we need padding?

Suppose we have:

```text
"I love cats"
```

and:

```text
"I love machine learning"
```

After tokenization:

```text
[10, 20, 30]
```

and:

```text
[10, 20, 40, 50]
```

Neural-network batches usually need tensors with consistent dimensions.

So we can pad:

```text
[10, 20, 30, PAD]
[10, 20, 40, 50]
```

Then we use an **attention mask** to tell the model:

```text
PAD = don't pay attention to this
```

For example:

```text
input IDs:

10  20  30  PAD
10  20  40   50

attention mask:

1   1   1   0
1   1   1   1
```

---

# 21. Attention mask vs tokenization

Don't confuse these.

Tokenization:

> What pieces of text exist?

Attention mask:

> Which positions should the model consider?

Example:

```text
Tokens:
["I", "love", "cats", "<PAD>"]

IDs:
[10, 20, 30, 0]

Mask:
[1, 1, 1, 0]
```

---

# 22. Truncation

LLMs have a maximum context length.

Suppose a model supports:

```text
4096 tokens
```

but your input contains:

```text
6000 tokens
```

You cannot simply feed all 6000.

You may need:

```text
6000 tokens
 ↓
truncate
 ↓
4096 tokens
```

In Hugging Face you'll encounter:

```python
truncation=True
```

For example:

```python
tokenizer(
    text,
    truncation=True,
    max_length=4096
)
```

---

# 23. Token count ≠ word count

This is extremely important for practical LLM work.

Suppose:

```text
Hello, how are you?
```

contains:

```text
4 words
```

but perhaps:

```text
6 tokens
```

depending on tokenizer.

Therefore:

```text
1 word ≠ 1 token
```

A rough English rule often used is around:

```text
1 token ≈ 0.75 English words
```

or:

```text
1 word ≈ 1.3 tokens
```

But this is only a rough approximation.

For code, other languages, unusual words, and multilingual text, token counts can differ substantially.

---

# 24. Why token count matters so much

Token count affects:

### Context window

If your model supports:

```text
128K tokens
```

you can provide a much larger context than with:

```text
8K tokens
```

### Computational cost

Transformers perform attention over sequences, and the sequence length strongly affects computation and memory.

### API cost

Many hosted LLM APIs charge based on input/output tokens.

### Training

Training datasets are often measured in:

```text
billions/trillions of tokens
```

rather than words.

### RAG

If you're building a RAG system, your chunks are ultimately converted into tokens before being sent to the model.

---

# 25. Why tokenization affects multilingual models

This is a fascinating practical issue.

Consider:

```text
Hello, how are you?
```

versus:

```text
नमस्ते, आप कैसे हैं?
```

A tokenizer may represent them using very different numbers of tokens.

Some languages can therefore require **more tokens to represent the same amount of information**.

This affects:

- context usage
- inference speed
- memory
- training efficiency
- API costs

This is one reason multilingual tokenizer design matters.

---

# 26. Tokenization of code

Tokenization isn't only for natural language.

Suppose you give an LLM:

```python
for i in range(10):
    print(i)
```

The tokenizer may produce pieces corresponding roughly to:

```text
for
 i
 in
 range
(
10
)
:
 print
(
i
)
```

But again, **don't assume those are the actual tokens**.

Programming syntax can create interesting tokenization behavior.

For example:

```python
get_user_information()
```

might be broken into several subword pieces.

This matters when you're using LLMs for coding.

---

# 27. Tokenization and context window

Suppose your model has:

```text
32,000 token context
```

You might think:

> "I can give it 32,000 words."

No.

It's:

```text
32,000 TOKENS
```

For example:

```text
PDF
 ↓
extract text
 ↓
text
 ↓
tokenizer
 ↓
tokens
```

If your PDF contains:

```text
50,000 tokens
```

it cannot necessarily fit into a 32K context window in one request.

That's why RAG systems often do:

```text
Large documents
      ↓
chunking
      ↓
embeddings
      ↓
vector database
      ↓
retrieve relevant chunks
      ↓
tokenize retrieved text
      ↓
LLM
```

---

# 28. Tokenization vs embeddings

These are two completely different things.

### Tokenization

Converts:

```text
text → token IDs
```

Example:

```text
"cat"
 ↓
[1234]
```

### Embedding

Converts:

```text
token ID → vector
```

Example:

```text
1234
 ↓
[0.13, -0.52, 0.81, ...]
```

So:

```text
TEXT
 ↓
TOKENIZER
 ↓
TOKEN IDs
 ↓
EMBEDDING TABLE
 ↓
VECTORS
 ↓
TRANSFORMER
```

Remember this pipeline.

---

# 29. Tokenization vs embeddings vs attention

These three concepts are often confused.

Think of them as three different stages:

```text
"I love AI"
     ↓
TOKENIZATION
     ↓
["I", " love", " AI"]
     ↓
TOKEN IDs
     ↓
[40, 3021, 9552]
     ↓
EMBEDDING
     ↓
vectors
     ↓
SELF-ATTENTION
     ↓
contextual representations
```

### Tokenization

"What pieces of text are here?"

### Embedding

"Represent each token as a vector."

### Attention

"How should each token interact with the other tokens?"

---

# 30. The most important practical example: Hugging Face

This is where your theoretical knowledge becomes useful.

Install:

```bash
pip install transformers
```

Then:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

Now:

```python
text = "I love machine learning."

tokens = tokenizer.tokenize(text)

print(tokens)
```

You may get something similar to:

```text
['i', 'love', 'machine', 'learning', '.']
```

Notice:

```text
tokens
```

are strings.

Now:

```python
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```

You'll get integers.

---

# 31. Even easier: tokenize directly

You can simply do:

```python
result = tokenizer("I love machine learning.")

print(result)
```

You'll get a structure containing things such as:

```python
{
    'input_ids': [...],
    'token_type_ids': [...],
    'attention_mask': [...]
}
```

Depending on the model, not all fields will appear.

---

# 32. `input_ids`

This is the most important field.

Example:

```python
input_ids:
[101, 1045, 2293, 3698, 4083, 1012, 102]
```

These are the token IDs.

Conceptually:

```text
"I love machine learning."
          ↓
[101, 1045, 2293, 3698, 4083, 1012, 102]
```

The model receives these IDs rather than the raw text.

---

# 33. `attention_mask`

Example:

```python
attention_mask:
[1, 1, 1, 1, 1, 1, 1]
```

If padding occurs:

```python
input_ids:
[101, 1045, 2293, 102, 0, 0]
```

then:

```python
attention_mask:
[1, 1, 1, 1, 0, 0]
```

The zeros indicate padding positions.

---

# 34. `tokenizer.decode()`

This is one of the most useful practical functions.

Suppose:

```python
ids = [101, 1045, 2293, 4083, 102]
```

You can convert them back:

```python
text = tokenizer.decode(ids)

print(text)
```

You might get:

```text
I love learning
```

So conceptually:

```text
TEXT
 ↓ encode
TOKEN IDs
 ↓ decode
TEXT
```

---

# 35. The complete Hugging Face workflow

A very useful mental model is:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

text = "I love machine learning."

# Text → tokens/IDs
encoded = tokenizer(text)

print(encoded["input_ids"])

# IDs → tokens
tokens = tokenizer.convert_ids_to_tokens(
    encoded["input_ids"]
)

print(tokens)

# IDs → text
decoded = tokenizer.decode(
    encoded["input_ids"]
)

print(decoded)
```

You should be comfortable doing this.

---

# 36. Batch tokenization

Real applications rarely process only one sentence.

Suppose:

```python
texts = [
    "I love AI.",
    "Transformers are powerful.",
    "Tokenization is important."
]
```

You can do:

```python
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
```

Now you have tensors suitable for PyTorch.

Conceptually:

```text
Text 1 ─┐
Text 2 ─┼→ Tokenizer → padded tensors
Text 3 ─┘
```

---

# 37. `return_tensors="pt"`

This tells Hugging Face:

> Return PyTorch tensors.

For example:

```python
{
    "input_ids": tensor(...),
    "attention_mask": tensor(...)
}
```

You can also encounter:

```python
return_tensors="tf"
```

for TensorFlow.

---

# 38. How tokenization works during LLM generation

Suppose you ask:

```text
What is Python?
```

The process roughly looks like:

```text
"What is Python?"
       ↓
Tokenizer
       ↓
[What, is, Python, ?]
       ↓
Token IDs
       ↓
Transformer
       ↓
probability distribution
       ↓
next token
```

Suppose the model predicts:

```text
"Python"
```

Then:

```text
"What is Python?"
+
" Python"
```

becomes the new sequence.

Then it predicts another token.

This happens repeatedly:

```text
input
 ↓
predict next token
 ↓
append token
 ↓
predict next token
 ↓
append token
 ↓
...
```

This is **autoregressive generation**.

---

# 39. This explains something very important

An LLM doesn't technically generate an entire sentence at once.

It generates:

> **one token at a time.**

Conceptually:

```text
The
```

→

```text
The capital
```

→

```text
The capital of
```

→

```text
The capital of France
```

→

```text
The capital of France is
```

and so on.

The exact internal generation process is more sophisticated, but this is the essential idea.

---

# 40. Why tokenization affects model behavior

Suppose a model sees:

```text
electrification
```

and its tokenizer represents it as:

```text
electr
ification
```

The model isn't processing the entire word as one atomic unit.

It processes those token pieces and learns relationships among them.

This is one reason LLMs can often handle words they've never explicitly seen during training.

---

# 41. A surprising consequence: tokenization can affect reasoning

Consider:

```text
123456789 × 987654321
```

The tokenizer may split the digits into various token pieces.

The model doesn't receive the integer as some special mathematical object.

It receives tokens.

This helps explain why LLMs can sometimes struggle with:

- unusual numbers
- long digit sequences
- arbitrary strings
- IDs
- hashes
- unfamiliar names

Tokenization can make the representation inconvenient for the neural network.

---

# 42. Why LLMs sometimes struggle with spelling

Suppose you ask:

```text
How many r's are in strawberry?
```

A model isn't necessarily internally manipulating the word as individual letters.

The tokenizer might represent:

```text
strawberry
```

as only a few tokens.

Therefore, asking it to count characters can be harder than you might expect.

This is an important consequence of subword tokenization.

---

# 43. Tokenization and prompt injection/security

Since you're interested in LLM security too, this becomes useful.

Security researchers sometimes investigate how strings are represented at the token level.

For example:

```text
normal text
```

and:

```text
weirdly constructed text
```

may tokenize differently even if they appear visually similar.

Unicode characters can also create surprising representations:

```text
A
А
```

These can look similar but be different Unicode characters.

Their tokenization can differ.

Therefore, tokenization is relevant to:

- prompt injection
- Unicode attacks
- jailbreak research
- input filtering
- token-based security systems

---

# 44. Tokenizer vocabulary size

A tokenizer might have a vocabulary containing tens of thousands or hundreds of thousands of tokens.

For example, conceptually:

```text
Vocabulary
────────────
token 0
token 1
token 2
...
token 100000
```

The model's embedding matrix has a corresponding number of rows.

If:

```text
vocab_size = 50,000
embedding_dimension = 4,096
```

then the embedding matrix is approximately:

```text
50,000 × 4,096
```

Every token gets an embedding vector.

This is one connection between:

**tokenizer → model architecture.**

---

# 45. Vocabulary size is a trade-off

Larger vocabulary:

### Advantages

Fewer tokens may be needed.

For example:

```text
"internationalization"
```

might fit into fewer tokens.

### Disadvantages

The model needs:

- a larger embedding matrix
- more vocabulary logits during output prediction
- more memory for vocabulary-related parameters

Smaller vocabulary:

### Advantages

Smaller vocabulary-related matrices.

### Disadvantages

Words may require more tokens.

So tokenizer vocabulary design involves trade-offs.

---

# 46. Why tokenizer and model must match

This is critical when using Hugging Face.

You shouldn't do:

```python
tokenizer = tokenizer_A
model = model_B
```

unless they are designed to work together.

Because the model's embedding matrix assumes a particular vocabulary.

For example:

```text
Tokenizer A:

"hello" → ID 1523
```

while:

```text
Tokenizer B:

"hello" → ID 847
```

If Model B expects its own ID mapping but you feed IDs from A, everything becomes incorrect.

Normally you do:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
```

using the matching model/tokenizer.

---

# 47. A practical experiment you should actually do

Create:

```python
from transformers import AutoTokenizer

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Tokenization is extremely important for LLMs!"

print("TEXT:")
print(text)

print("\nTOKENS:")
tokens = tokenizer.tokenize(text)
print(tokens)

print("\nTOKEN IDs:")
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

print("\nDECODED:")
print(tokenizer.decode(ids))
```

Then try:

```text
Hello world
```

```text
machine learning
```

```text
transformers
```

```text
unbelievable
```

```text
supercalifragilisticexpialidocious
```

```text
नमस्ते दुनिया
```

```text
こんにちは
```

```text
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
```

You'll start developing an intuitive understanding of tokenization.

---

# 48. An even better experiment

Print everything:

```python
text = "I love machine learning!"

encoded = tokenizer(
    text,
    return_tensors="pt"
)

print("Input IDs:")
print(encoded["input_ids"])

print("\nAttention Mask:")
print(encoded["attention_mask"])

print("\nTokens:")

for token_id in encoded["input_ids"][0]:
    token_id = token_id.item()
    token = tokenizer.convert_ids_to_tokens(token_id)

    print(token_id, "→", repr(token))
```

You'll literally see:

```text
101 → '[CLS]'
1045 → 'i'
2293 → 'love'
...
102 → '[SEP]'
```

depending on the tokenizer.

This is one of the best ways to understand it.

---

# 49. One important distinction: tokenizer training vs tokenizer usage

There are two different activities.

## Tokenizer training

You create a tokenizer from a corpus.

```text
Huge text dataset
       ↓
learn frequent patterns
       ↓
build vocabulary
       ↓
tokenizer
```

This happens when developing a model.

## Tokenizer usage

You take an existing tokenizer:

```python
AutoTokenizer.from_pretrained(...)
```

and use it:

```text
new text
 ↓
existing tokenizer
 ↓
token IDs
```

As someone learning LLM application development, you'll mostly be doing the second.

---

# 50. What happens when fine-tuning?

This is particularly relevant to your Hugging Face goals.

Suppose you're fine-tuning a model.

Your dataset:

```text
Question: What is Docker?
Answer: Docker is a containerization platform...
```

First:

```text
dataset text
 ↓
tokenizer
 ↓
input IDs
```

Then:

```text
input IDs
 ↓
model
 ↓
predictions
 ↓
loss
 ↓
backpropagation
 ↓
weight updates
```

So tokenization is literally part of your training pipeline.

---

# 51. Tokenization during LoRA fine-tuning

If you're doing LoRA fine-tuning, the flow is still:

```text
Dataset
   ↓
Tokenizer
   ↓
Input IDs
   ↓
Base LLM
   +
LoRA adapters
   ↓
Predictions
   ↓
Loss
   ↓
Backpropagation
   ↓
LoRA weights updated
```

LoRA does **not** replace tokenization.

---

# 52. Tokenization in RAG

Your planned RAG systems will also rely heavily on this.

Suppose you have a PDF:

```text
GATE CSE Operating Systems notes
```

Pipeline:

```text
PDF
 ↓
extract text
 ↓
chunk
 ↓
embedding model
 ↓
vector database
```

Then a user asks:

```text
What is virtual memory?
```

The query is tokenized by the embedding model.

Retrieved chunks are then eventually tokenized by the LLM:

```text
Question
   +
Retrieved context
        ↓
    Tokenizer
        ↓
    Token IDs
        ↓
       LLM
```

Therefore:

> Tokenization is present in almost every stage of an LLM application, although different models may use different tokenizers.

---

# 53. Chunk size vs token size

This will become important when you build RAG.

Suppose you say:

```text
chunk_size = 500
```

What does 500 mean?

It could mean:

```text
500 characters
```

or:

```text
500 words
```

or:

```text
500 tokens
```

These are completely different.

For LLM applications, **token-based chunking** can often be more meaningful because the model's context limit is measured in tokens.

---

# 54. The complete picture

Now connect everything you've learned about Transformers.

```text
                 HUMAN
                   │
                   ▼
             Raw text
                   │
                   ▼
              TOKENIZER
                   │
                   ▼
                Tokens
                   │
                   ▼
              Token IDs
                   │
                   ▼
          Embedding lookup
                   │
                   ▼
              Embeddings
                   │
                   ▼
        Positional information
                   │
                   ▼
        ┌───────────────────┐
        │     TRANSFORMER   │
        │                   │
        │ Self-Attention    │
        │       ↓           │
        │ Feed Forward      │
        │       ↓           │
        │ Self-Attention    │
        │       ↓           │
        │ ...               │
        └───────────────────┘
                   │
                   ▼
          Output probabilities
                   │
                   ▼
           Next-token choice
                   │
                   ▼
             New token
                   │
                   └──────────┐
                              │
                              ▼
                         Repeat...
```

That is the core LLM loop.

---

# 55. The 10 things you should remember

If you forget everything else, remember these:

### 1.

**LLMs don't directly read text.**

They process numbers.

### 2.

**Tokenization converts text into tokens.**

```text
Text → Tokens
```

### 3.

**Tokens aren't necessarily words.**

They can be:

```text
words
subwords
characters
bytes
punctuation
spaces
special tokens
```

### 4.

**Token IDs are numerical identifiers.**

```text
token → ID
```

### 5.

**IDs themselves don't contain meaning.**

The embedding layer gives tokens useful vector representations.

### 6.

**Different models can have different tokenizers.**

Therefore token IDs are model/tokenizer-specific.

### 7.

**Modern LLMs commonly use subword/byte-level approaches.**

Important algorithms include:

```text
BPE
WordPiece
Unigram
```

### 8.

**Token count matters.**

It affects:

```text
context length
memory
computation
training
API cost
RAG chunking
```

### 9.

**Tokenizer and model must normally match.**

```text
Tokenizer ↔ Model
```

### 10.

**LLMs generate tokens, usually one token at a time.**

```text
context
 ↓
predict next token
 ↓
append
 ↓
predict next token
 ↓
...
```

---

# 56. Your practical mental model

Whenever you see an LLM application, imagine this:

```text
                USER
                  │
                  ▼
             "Hello AI!"
                  │
                  ▼
             TOKENIZER
                  │
                  ▼
       ["Hello", " AI", "!"]
                  │
                  ▼
          [15496, 9552, 0]
                  │
                  ▼
             EMBEDDINGS
                  │
                  ▼
          ┌───────────────┐
          │  TRANSFORMER  │
          │               │
          │ Attention     │
          │      ↓        │
          │ FFN           │
          │      ↓        │
          │ Attention     │
          │      ↓        │
          │ ...           │
          └───────────────┘
                  │
                  ▼
       Probability of next token
                  │
                  ▼
            " How"
                  │
                  ▼
             repeat...
```

Once this picture is clear, **tokenization stops being an isolated NLP topic and becomes the first stage of the entire LLM pipeline**.

### What I'd learn immediately after this

Given your goal of becoming practically capable with LLMs, the natural sequence is:

```text
1. Tokenization               ← you are here
        ↓
2. Embeddings
        ↓
3. Attention / Self-Attention
        ↓
4. Transformer architecture
        ↓
5. Positional encoding
        ↓
6. Encoder vs Decoder
        ↓
7. Causal language modeling
        ↓
8. Pretraining
        ↓
9. Fine-tuning
        ↓
10. LoRA / PEFT
        ↓
11. Quantization
        ↓
12. RAG
        ↓
13. Hugging Face Transformers
        ↓
14. Build/deploy your own LLM applications
```

The **most important next concept is embeddings**, because you now know how `"machine learning"` becomes token IDs; the next question is **how those IDs become vectors that a Transformer can actually reason over**.

