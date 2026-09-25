# embeddings practically

# Embeddings — the complete practical explanation

## 1. First: what problem do embeddings solve?

Computers fundamentally work with numbers.

Suppose we have:

> `cat`

A neural network cannot directly process the meaning of the word `"cat"`.

We first tokenize it:

```text
"cat" → token ID  →  4821
```

But `4821` is still not meaningful mathematically.

For example:

```text
cat → 4821
dog → 9132
car → 1742
banana → 6288
```

The numbers are essentially **IDs**.

There is no reason why:

```text
9132 > 4821
```

should mean that dog is somehow "more" than cat.

We therefore need a representation where **relationships between concepts can be represented mathematically**.

That's what an embedding does.

> **An embedding is a vector of numbers that represents an object in a continuous mathematical space such that useful relationships can be learned from distances/directions in that space.**

For example, conceptually:

```text
cat     → [0.21, -0.43, 0.78, ...]
dog     → [0.18, -0.39, 0.75, ...]
car     → [-0.72, 0.11, -0.31, ...]
```

The actual values are learned by the neural network.

---

# 2. What exactly is a vector?

Before embeddings, you need to be comfortable with vectors.

A vector is simply an ordered list of numbers.

For example:

```text
[2, 5, 1]
```

is a 3-dimensional vector.

An embedding might look like:

```text
[-0.134, 0.721, 0.092, -0.551, ...]
```

If it has 768 numbers:

```text
embedding dimension = 768
```

If it has 1536 numbers:

```text
embedding dimension = 1536
```

So:

```text
word
 ↓
embedding
 ↓
[0.12, -0.83, 0.42, ..., 0.17]
```

The vector is the machine-learning representation of the object.

---

# 3. Why not just use token IDs?

Consider:

```text
cat → 10
dog → 11
car → 12
banana → 13
```

The model could incorrectly interpret these as numerical quantities.

For example:

```text
dog - cat = 1
car - dog = 1
```

But token IDs don't encode semantic relationships.

Instead we want something like:

```text
cat → [0.8, 0.7, 0.1, ...]
dog → [0.79, 0.72, 0.08, ...]

car → [-0.2, 0.1, 0.9, ...]
```

Now `cat` and `dog` can end up near each other because they're semantically related.

That's the fundamental idea.

---

# 4. Embeddings create a "space"

This is probably the most important mental model.

Imagine a huge map.

Each concept is represented as a point.

For a simplified 2D example:

```text
             animals

       dog ●
           \
            ● cat

● banana
                              \
                               ● apple

● car
       \
        ● bus
```

In reality, embeddings aren't usually 2D.

They might have:

```text
384 dimensions
768 dimensions
1024 dimensions
1536 dimensions
3072 dimensions
```

You cannot visualize that directly.

But mathematically, it's still just a coordinate system.

---

# 5. What does "meaning" look like mathematically?

Suppose:

```text
cat = [0.8, 0.7]
dog = [0.75, 0.72]
car = [-0.3, 0.1]
```

Cat and dog are close.

Car is farther away.

We can calculate their distance.

For two vectors:

```text
A = [a₁, a₂]
B = [b₁, b₂]
```

Euclidean distance is:

$$
d(A,B)=\sqrt{(a_1-b_1)^2+(a_2-b_2)^2}
$$

So embeddings allow us to turn:

> "Are these things similar?"

into:

> "How close are these vectors?"

That's incredibly useful.

---

# 6. But embeddings aren't simply "meaning vectors"

This is an important correction.

You might hear:

> "Each dimension represents something like intelligence, gender, animalness, etc."

Usually you **shouldn't think about embeddings that literally**.

For example, dimension 57 doesn't necessarily mean:

```text
dimension 57 = animalness
```

The representation is distributed across many dimensions.

Meaning is encoded through patterns across the entire vector.

Think:

```text
Embedding
┌───────────────────────────────┐
│ .23 -.72 .11 .84 -.19 ... .42 │
└───────────────────────────────┘
          ↑
    distributed representation
```

Not:

```text
dimension 1 = gender
dimension 2 = size
dimension 3 = animal
```

---

# 7. Where do embeddings come from?

This is where embeddings connect directly to Transformers.

Suppose your tokenizer produces:

```text
I love cats
```

Tokens:

```text
["I", "love", "cats"]
```

Token IDs:

```text
[42, 1876, 9214]
```

The model has an **embedding matrix**.

Suppose, for simplicity:

```text
Vocabulary = 10,000 tokens
Embedding dimension = 4
```

The embedding matrix is:

$$
E \in \mathbb{R}^{10000 \times 4}
$$

Conceptually:

```text
             dimension
          1     2     3     4

token 0  [ ...  ...  ...  ... ]
token 1  [ ...  ...  ...  ... ]
token 2  [ ...  ...  ...  ... ]
...
token 42 [ .12  .83 -.41  .27 ]
...
token 1876 [...]
...
token 9214 [...]
```

When the input contains token `42`, the model retrieves row 42.

So:

```text
token ID
   ↓
embedding matrix
   ↓
corresponding row
   ↓
vector
```

---

# 8. Embedding lookup

This is extremely important practically.

Suppose:

```text
Embedding matrix E:

E =
[
 [0.1, 0.2, 0.3],
 [0.4, 0.5, 0.6],
 [0.7, 0.8, 0.9],
 [1.0, 1.1, 1.2]
]
```

Suppose token IDs are:

```text
[2, 0, 3]
```

The model retrieves:

```text
token 2 → [0.7, 0.8, 0.9]

token 0 → [0.1, 0.2, 0.3]

token 3 → [1.0, 1.1, 1.2]
```

Therefore:

```text
input:

[2, 0, 3]

        ↓

[
 [0.7, 0.8, 0.9],
 [0.1, 0.2, 0.3],
 [1.0, 1.1, 1.2]
]
```

Now the Transformer has numerical vectors to work with.

---

# 9. Embedding layer vs embedding vector

These terms are often confusing.

### Embedding layer

The entire trainable lookup table:

```text
Vocabulary × embedding_dimension
```

Example:

```text
50,000 × 768
```

### Embedding vector

One row from that table:

```text
768 numbers
```

For example:

```text
"cat"
   ↓
[0.21, -0.31, 0.92, ...]
```

So:

```text
Embedding layer
       ↓
   lookup token
       ↓
Embedding vector
```

---

# 10. Are embeddings learned?

**Yes.**

This is one of the most important ideas.

Suppose initially:

```text
cat → random vector
dog → random vector
car → random vector
```

Something like:

```text
cat → [0.13, -0.72, 0.44]
dog → [-0.52, 0.11, 0.81]
car → [0.33, 0.41, -0.27]
```

During training, the neural network makes predictions.

Suppose it makes a mistake.

The loss function measures the error:

```text
prediction
    ↓
loss
    ↓
backpropagation
    ↓
gradients
    ↓
update embedding values
```

Repeated billions of times, the embeddings become useful representations.

So embeddings aren't manually designed.

They're **learned parameters**.

---

# 11. A very important distinction: token embedding vs contextual representation

This is where many beginners get confused.

Consider:

> "I went to the bank to deposit money."

and:

> "I sat beside the river bank."

The token:

```text
bank
```

starts with the same token embedding.

But its meaning in the two sentences is different.

Modern Transformers solve this through contextual processing.

Initially:

```text
bank
 ↓
same token embedding
```

Then attention and Transformer layers process the surrounding tokens:

```text
"I went to the bank to deposit money"

                    ↓

contextual representation of "bank"
                    ↓
              financial meaning
```

Whereas:

```text
"I sat beside the river bank"

                    ↓

contextual representation of "bank"
                    ↓
              riverside meaning
```

So distinguish:

### Token embedding

Initial representation associated with the token.

### Contextual representation

Representation after the Transformer has processed context.

This distinction is **very important for LLMs**.

---

# 12. Embeddings + positional information

There's another problem.

Suppose:

```text
Dog bites man
```

and:

```text
Man bites dog
```

Same words.

Different meaning.

If we only give the Transformer token embeddings:

```text
Dog → vector
bites → vector
man → vector
```

the model needs to know **where each token occurs**.

That's why Transformers add positional information.

Conceptually:

```text
Token embedding
       +
Position information
       ↓
Transformer input
```

Depending on the architecture, this may use positional embeddings, RoPE, or another positional mechanism.

So the early Transformer pipeline looks roughly like:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Token embedding
 ↓
Position information
 ↓
Transformer
 ↓
Contextual representations
 ↓
Q / K / V attention
 ↓
...
```

---

# 13. How embeddings relate to Q, K and V

Since you've just been learning Q/K/V, this connection is extremely important.

Suppose your embedding/input representation is:

$$
X
$$

The Transformer creates:

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

So embeddings are essentially the **starting representation** from which the attention mechanism constructs Q, K and V.

Conceptually:

```text
Token
 ↓
Embedding
 ↓
X
 ├── × WQ → Q
 ├── × WK → K
 └── × WV → V
```

Then:

$$
Attention(Q,K,V)
=
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

So you can now see the relationship:

```text
Tokenization
     ↓
Token IDs
     ↓
Embeddings
     ↓
X
     ↓
Q / K / V
     ↓
Attention
     ↓
Transformer layers
     ↓
Contextual representation
```

---

# 14. Embeddings aren't only for words

This is a huge concept.

You can embed almost anything.

For example:

### Text

```text
sentence → vector
```

### Images

```text
image → vector
```

### Audio

```text
audio → vector
```

### Products

```text
product → vector
```

### Users

```text
user → vector
```

### Documents

```text
document → vector
```

The general idea is:

> Convert an object into a numerical representation that captures useful relationships.

---

# 15. Sentence embeddings

Instead of embedding individual tokens, we can create an embedding for an entire sentence.

Example:

```text
"The cat is sleeping."

        ↓

[0.12, -0.72, 0.44, ..., 0.31]
```

And:

```text
"A kitten is asleep."

        ↓

[0.11, -0.70, 0.42, ..., 0.29]
```

These vectors can be close because the sentences have similar meanings.

This is called a **sentence embedding**.

---

# 16. Document embeddings

Same idea, but for larger pieces of text.

For example:

```text
KIIT examination regulations.pdf

             ↓

       embedding vector
```

Then another document:

```text
KIIT attendance rules.pdf

             ↓

       embedding vector
```

Documents discussing similar topics can have similar representations.

This leads directly to **RAG**.

---

# 17. Embeddings are the foundation of semantic search

Traditional keyword search asks:

> Does this document contain the same words?

Suppose the user asks:

> "How many classes do I need to attend?"

A document says:

> "Students must maintain a minimum attendance percentage."

It might not contain:

```text
"how many classes"
```

but the meanings are related.

Embedding search can detect this.

Conceptually:

```text
User query
   ↓
embedding
   ↓
q = [0.21, -0.72, ...]

Documents
   ↓
embeddings

doc1 = [0.19, -0.70, ...]
doc2 = [-0.71, 0.12, ...]
doc3 = [0.22, -0.68, ...]
```

Similarity:

```text
query ↔ doc1 → high
query ↔ doc2 → low
query ↔ doc3 → high
```

Retrieve the most similar chunks.

---

# 18. This is exactly how a basic RAG system works

Suppose you have 50 PDFs.

You don't want to put all 50 PDFs into every prompt.

Instead:

### Step 1 — Split documents

```text
PDF
 ↓
chunks

chunk 1
chunk 2
chunk 3
...
```

### Step 2 — Embed every chunk

```text
chunk 1 → vector
chunk 2 → vector
chunk 3 → vector
```

### Step 3 — Store vectors

Usually in a vector database/index.

Examples include systems such as:

```text
FAISS
Qdrant
Milvus
pgvector
Chroma
```

### Step 4 — User asks question

```text
"What is the attendance requirement?"
```

### Step 5 — Embed the question

```text
question → query vector
```

### Step 6 — Search nearest vectors

```text
query vector
      ↓
similarity search
      ↓
top relevant chunks
```

### Step 7 — Give those chunks to the LLM

```text
Question
+
Retrieved context
        ↓
       LLM
        ↓
      Answer
```

That's RAG.

---

# 19. How do we measure similarity?

One of the most common methods is **cosine similarity**.

Given vectors $A$ and $B$:

$$
\cos(\theta)
=
\frac{A\cdot B}
{\|A\|\|B\|}
$$

where:

$$
A\cdot B
$$

is the dot product.

The intuition is:

```text
same direction → high similarity

different direction → low similarity
```

For normalized vectors:

```text
cosine similarity ≈ dot product
```

This is why vector databases often perform similarity searches using inner product/cosine-style metrics.

---

# 20. Example of cosine similarity

Suppose:

```text
A = [1, 0]

B = [0.9, 0.1]
```

These vectors point almost in the same direction.

Therefore:

```text
similarity ≈ 0.99
```

But:

```text
C = [-1, 0]
```

points in the opposite direction.

Therefore:

```text
similarity = -1
```

So:

```text
A ↔ B = very similar direction

A ↔ C = opposite direction
```

---

# 21. Why does similarity correspond to meaning?

This is an important conceptual question.

Nobody manually tells the model:

```text
cat should be near dog
```

Instead, training creates statistical pressure.

Words occurring in similar contexts tend to develop related representations.

For example:

```text
The cat eats fish.
The dog eats fish.
The cat plays outside.
The dog plays outside.
```

The model repeatedly sees relationships between these concepts.

Training adjusts parameters so that representations become useful for prediction.

Over enormous datasets, semantic structure emerges.

---

# 22. The classic intuition: "You shall know a word by the company it keeps"

Consider:

```text
The cat drinks milk.
The cat sleeps.
The cat catches mice.
```

and:

```text
The dog drinks water.
The dog sleeps.
The dog catches balls.
```

Cat and dog appear in similar contexts.

Therefore their learned representations can become similar.

This general principle is one of the foundations behind distributional representations.

---

# 23. Word embeddings vs modern LLM embeddings

Historically, you may encounter:

### Word2Vec

Creates word vectors.

```text
king → vector
queen → vector
man → vector
woman → vector
```

### GloVe

Another classic word-embedding technique.

### FastText

Uses subword information.

These are generally **static embeddings**.

The word:

```text
bank
```

has essentially one vector.

Modern Transformers produce contextual representations:

```text
bank + context
        ↓
contextual representation
```

This is much more powerful.

---

# 24. The famous vector arithmetic example

You may see:

$$
king - man + woman \approx queen
$$

This is a famous observation from word embeddings.

The intuition is:

```text
king
  -
man
  +
woman
  ≈
queen
```

But don't treat this as a universal law.

Modern embeddings are much more complicated, and vector arithmetic doesn't reliably work for arbitrary concepts.

The important lesson is:

> Semantic relationships can sometimes appear as geometric relationships in embedding space.

---

# 25. Embedding dimension

Suppose your embedding has:

```text
768 dimensions
```

Then every object is represented by:

```text
768 floating-point values
```

Example:

```text
[-0.12,
  0.42,
 -0.73,
  0.11,
  ...
  0.09]
```

Why so many dimensions?

Because language contains enormous amounts of information.

A higher-dimensional space gives the model more capacity to represent complicated relationships.

But:

> Bigger dimension does not automatically mean better embedding.

The quality depends on the model, training data, training objective, and use case.

---

# 26. Embedding size vs LLM hidden size

These terms can be confusing.

An LLM might have:

```text
hidden size = 4096
```

while a separate sentence-embedding model might output:

```text
embedding dimension = 768
```

They are not necessarily the same thing.

For example:

```text
LLM internal representation
        ↓
4096-dimensional

Embedding model output
        ↓
768-dimensional
```

An embedding model may be specifically trained to produce vectors suitable for similarity/search.

---

# 27. LLM embeddings and RAG embeddings are not necessarily the same

This is extremely important for your practical LLM work.

You might have:

```text
Qwen / Llama / Mistral
```

for **generation**.

And separately:

```text
BGE
E5
Nomic
etc.
```

for **embeddings**.

So a RAG system can look like:

```text
             ┌──────────────┐
             │ Embedding    │
             │ model        │
             └──────┬───────┘
                    │
             vectors/search
                    │
                    ↓
User → Retriever → relevant chunks
                    │
                    ↓
              ┌───────────┐
              │ LLM       │
              │ Qwen etc. │
              └───────────┘
                    ↓
                 Answer
```

The embedding model and generation model can be completely different models.

---

# 28. Embedding model vs generative LLM

Think of their jobs as different.

### Embedding model

Input:

```text
"What is Docker?"
```

Output:

```text
[0.12, -0.43, 0.91, ...]
```

Its job is to represent meaning.

### Generative LLM

Input:

```text
"What is Docker?"
```

Output:

```text
"Docker is a platform..."
```

Its job is to generate text.

So:

```text
Embedding model → representation

LLM → generation
```

---

# 29. What does a vector database actually store?

Suppose you have:

```text
chunk_id: 17

text:
"Docker containers package applications..."

embedding:
[0.12, -0.51, 0.72, ...]
```

The vector database may associate:

```text
ID
 ↓
vector
 ↓
metadata
 ↓
original chunk
```

For example:

```text
ID: 17

vector:
[0.12, -0.51, ...]

metadata:
{
  "file": "docker.pdf",
  "page": 12
}

text:
"Docker containers package..."
```

Then you can search by vector.

---

# 30. Chunking matters enormously

Suppose you have a 100-page PDF.

Don't necessarily embed the entire PDF as one vector.

Instead:

```text
PDF
 ↓
chunk 1
chunk 2
chunk 3
...
chunk 500
```

Then:

```text
chunk 1 → embedding 1
chunk 2 → embedding 2
...
chunk 500 → embedding 500
```

When the user asks something, retrieve the relevant chunks.

This is why embeddings are central to your plan of working with lots of PDFs locally.

---

# 31. Metadata is different from embeddings

Suppose:

```text
chunk:
"Attendance must be above 75%."

embedding:
[0.13, -0.82, ...]
```

The embedding represents semantic information.

Metadata might be:

```json
{
  "file": "regulations.pdf",
  "page": 27,
  "course": "CSE",
  "year": 2026
}
```

Metadata lets you perform filters.

For example:

```text
semantic similarity
+
course = CSE
+
year = 2026
```

This is often better than relying only on embeddings.

---

# 32. Embeddings can represent images too

Imagine:

```text
image of a dog
```

An image encoder can produce:

```text
[0.17, -0.31, 0.82, ...]
```

A text encoder could produce:

```text
"a dog playing in a park"
        ↓
[0.19, -0.29, 0.79, ...]
```

If the models are trained into a shared embedding space, those vectors can be close.

This is the idea behind multimodal systems such as CLIP-like approaches.

---

# 33. Multimodal embeddings

You can potentially have:

```text
Text
 ↓
embedding

Image
 ↓
embedding

Audio
 ↓
embedding
```

If they're aligned into a shared space:

```text
        Shared vector space

       image of dog ●
                    \
                     ● "dog running"
                    /
         ● dog audio
```

This allows cross-modal retrieval.

For example:

> Search images using natural-language text.

---

# 34. Embeddings are learned through objectives

This is a deeper but important concept.

Different embedding models are trained for different goals.

For example:

### Classification objective

Make representation useful for predicting classes.

### Contrastive objective

Make related pairs close:

```text
positive pair
     ↓
closer
```

and unrelated pairs farther:

```text
negative pair
     ↓
farther
```

For example:

```text
"The dog is running."
"A dog runs outside."
```

Positive pair.

Whereas:

```text
"The dog is running."
"Quantum mechanics explains..."
```

could be a negative pair depending on the training dataset/objective.

---

# 35. Contrastive learning intuition

Imagine:

```text
Query: "How to install Docker?"

Positive:
"Docker installation instructions"

Negative:
"History of operating systems"
```

Training tries to produce:

```text
distance(query, positive) → small

distance(query, negative) → large
```

Repeated over many examples, the embedding space becomes useful for retrieval.

This is one reason modern embedding models can be excellent for semantic search.

---

# 36. Why embeddings can fail

Embeddings are not magic.

Suppose:

```text
Apple
```

Could mean:

```text
fruit
```

or:

```text
company
```

The representation depends on the model and context.

Poor chunking can also cause problems.

For example:

```text
chunk 1:
"The exam is conducted..."

chunk 2:
"...in December."

```

Separately, neither chunk may contain enough information.

Good RAG therefore requires:

```text
good embedding model
+
good chunking
+
good retrieval
+
good metadata
+
good prompting
```

---

# 37. Semantic similarity isn't factual correctness

This is another extremely important distinction.

Suppose the query is:

> "What is the deadline?"

The embedding system might retrieve:

> "The registration process begins on September 1."

because the concepts are related.

But that doesn't mean it's the correct answer.

Embeddings answer:

> "What is semantically similar?"

They don't inherently answer:

> "What is factually correct?"

That's why retrieval systems need good indexing, metadata, reranking, and generation.

---

# 38. Embedding search vs keyword search

Imagine the query:

> "How can I put an application inside a container?"

Document:

> "Docker packages applications into isolated containers."

Keyword search might struggle because:

```text
put application
```

doesn't exactly match:

```text
packages applications
```

Embedding search can recognize the semantic relationship.

So:

```text
Keyword search
→ lexical similarity

Embedding search
→ semantic similarity
```

Modern retrieval systems often combine both.

---

# 39. Dense vs sparse representations

Embeddings are generally called **dense vectors**.

Example:

```text
[0.23, -0.41, 0.72, 0.09, ...]
```

Most values are non-zero.

Sparse representations contain mostly zeros.

Example:

```text
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, ...]
```

Traditional bag-of-words/TF-IDF representations are sparse.

Modern semantic embeddings are generally dense.

---

# 40. Embedding pipeline you should memorize

This is the practical pipeline:

```text
             TEXT
               │
               ▼
          Tokenization
               │
               ▼
           Token IDs
               │
               ▼
        Embedding lookup
               │
               ▼
       Token embeddings
               │
               ▼
       Position information
               │
               ▼
        Transformer layers
               │
               ▼
    Contextual representations
               │
               ▼
       ┌───────────────┐
       │               │
       ▼               ▼
   Generation       Representation
                       │
                       ▼
                 Embedding/search
```

Don't confuse the two uses.

---

# 41. Practical RAG pipeline

For the kind of local AI system you're interested in, memorize this:

```text
                  DOCUMENTS
                      │
                      ▼
                   Chunking
                      │
                      ▼
               Embedding model
                      │
                      ▼
                  Vectors
                      │
                      ▼
              Vector database
                      │
                      │
User question ───────┤
      │               │
      ▼               │
Embedding model       │
      │               │
      ▼               │
Query vector ─────────┘
                      │
                      ▼
               Similarity search
                      │
                      ▼
               Relevant chunks
                      │
                      ▼
                 Prompt + chunks
                      │
                      ▼
                     LLM
                      │
                      ▼
                   Answer
```

This is probably the **single most useful practical application of embeddings for you**.

---

# 42. Let's actually implement one

The theory becomes much easier once you see the code.

A simple conceptual Python example:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat is sleeping.",
    "A kitten is asleep.",
    "Docker is used for containers."
]

embeddings = model.encode(sentences)

print(embeddings.shape)
```

You might get something like:

```text
(3, 384)
```

Meaning:

```text
3 sentences
×
384-dimensional vectors
```

So:

```text
sentence 1 → 384 numbers
sentence 2 → 384 numbers
sentence 3 → 384 numbers
```

---

# 43. Calculate similarity

For example:

```python
from sentence_transformers import util

similarity = util.cos_sim(
    embeddings[0],
    embeddings[1]
)

print(similarity)
```

You would expect:

```text
"The cat is sleeping."
```

and:

```text
"A kitten is asleep."
```

to have relatively high semantic similarity.

Whereas:

```text
"The cat is sleeping."
```

and:

```text
"Docker is used for containers."
```

should generally be much less similar.

---

# 44. Your first tiny semantic search engine

You can build one yourself:

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Docker packages applications into containers.",
    "Python is a programming language.",
    "Kubernetes manages containerized applications.",
    "Git is a version control system."
]

doc_embeddings = model.encode(documents)

query = "How do I package my application in a container?"

query_embedding = model.encode(query)

scores = util.cos_sim(
    query_embedding,
    doc_embeddings
)[0]

for i, score in enumerate(scores):
    print(score.item(), documents[i])
```

The highest-scoring documents should generally be the ones semantically related to the query.

You've now built the core of semantic retrieval.

---

# 45. What happens inside `model.encode()`?

At a high level:

```text
text
 ↓
tokenizer
 ↓
token IDs
 ↓
Transformer
 ↓
contextual token representations
 ↓
pooling
 ↓
sentence embedding
```

The exact architecture depends on the embedding model.

This is an important distinction:

```text
LLM:
text → tokens → Transformer → next-token probabilities
```

Embedding model:

```text
text → tokens → Transformer → vector
```

Same broad neural-network concepts, different objective/output.

---

# 46. Pooling

Suppose the Transformer produces a vector for every token:

```text
"The cat sleeps"

The     → vector
cat     → vector
sleeps  → vector
```

But we want one vector representing the entire sentence.

We need some method of combining the token representations.

This is called **pooling**.

For example:

```text
token vectors
     ↓
pooling
     ↓
one sentence vector
```

Possible methods include:

- mean pooling
- CLS pooling
- max pooling
- learned pooling

The correct method depends on the model architecture/training.

---

# 47. Don't confuse three things

This is one of the biggest sources of confusion:

### 1. Token ID

```text
cat → 4821
```

Just an identifier.

### 2. Token embedding

```text
cat → [0.12, -0.43, ...]
```

Learned vector from the embedding layer.

### 3. Contextual representation

```text
cat + surrounding context
→ [0.91, -0.12, ...]
```

Representation after Transformer processing.

And sometimes:

### 4. Sentence/document embedding

```text
whole sentence/document
→ one vector
```

Usually generated specifically for retrieval/similarity.

---

# 48. Embeddings in fine-tuning

Embeddings also matter during LLM training.

The embedding matrix itself can be trainable.

During backpropagation:

```text
prediction
 ↓
loss
 ↓
gradient
 ↓
embedding parameters updated
```

So when you fine-tune a model, depending on the method and configuration, the token embedding parameters may or may not be updated.

With LoRA, most base model parameters—including usually the embedding matrix—remain frozen while low-rank adapter parameters are trained.

That's why understanding embeddings helps you understand fine-tuning.

---

# 49. Embeddings in your local LLM setup

Suppose you're building your local AI assistant.

You might have:

```text
Qwen / Llama
      +
embedding model
      +
Qdrant/FAISS
      +
RAG
```

For example:

```text
             Your PDFs
                 │
                 ▼
              Chunking
                 │
                 ▼
         Local embedding model
                 │
                 ▼
              Vector DB
                 │
                 │
Question ────────┘
   │
   ▼
embedding
   │
   ▼
retrieve relevant chunks
   │
   ▼
Qwen/Llama
   │
   ▼
answer
```

The LLM does **not need to memorize all your PDFs**.

The embedding system helps it find the relevant information when needed.

---

# 50. Embedding models you will encounter

You'll likely see families such as:

- BGE
- E5
- Nomic Embed
- Sentence-Transformers
- GTE

The important thing isn't memorizing model names.

Look at:

```text
embedding dimension
context length
language support
model size
quality
latency
hardware requirements
license
query/document instruction format
```

For a local setup, model size and inference speed matter.

---

# 51. One subtle but important issue: query/document embeddings

Some embedding models expect different prefixes/instructions for queries and documents.

Conceptually:

```text
query:
"query: What is Docker?"

document:
"passage: Docker is..."
```

This is model-dependent.

You should **always check the embedding model's documentation** rather than assuming you can encode everything identically.

---

# 52. Normalization

You may encounter:

```python
normalize_embeddings=True
```

This normalizes vectors so their magnitude becomes approximately 1.

For a vector:

$$
v=[v_1,v_2,\ldots,v_n]
$$

its norm is:

$$
\|v\|=\sqrt{\sum_i v_i^2}
$$

Normalized:

$$
v'=\frac{v}{\|v\|}
$$

Then:

```text
||v'|| ≈ 1
```

This can make cosine similarity efficiently implemented using dot products.

Again, follow the model/vector database's recommended configuration.

---

# 53. Embedding dimensions and storage

Suppose you have:

```text
1,000,000 chunks
```

and each embedding has:

```text
1536 dimensions
```

with float32 values.

Approximately:

$$
1,000,000 \times 1536 \times 4
$$

bytes.

That's about:

```text
6.14 GB
```

just for raw vectors, before indexing/metadata/database overhead.

This is why embedding dimension, quantization, and index design matter at scale.

For your personal/local RAG system with perhaps thousands or tens of thousands of chunks, this is much less concerning.

---

# 54. What embeddings DON'T do

Embeddings don't automatically:

❌ generate answers  
❌ understand truth perfectly  
❌ replace an LLM  
❌ guarantee correct retrieval  
❌ remember information permanently  
❌ perform reasoning by themselves  
❌ magically understand every language equally well  

Their primary job is:

> **Represent information in a vector space where useful relationships can be measured.**

---

# 55. The deepest mental model

If you remember only one conceptual picture, remember this:

```text
                  REAL WORLD
                      │
                      ▼
              ┌───────────────┐
              │ Neural network│
              └───────┬───────┘
                      │
                      ▼
              VECTOR SPACE
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
     cat            dog            car
      ●              ●              ●

   similar         similar        different
   concepts        concepts       concept
```

The model transforms objects into points in a mathematical space.

Then operations such as:

```text
distance
similarity
nearest neighbor
clustering
classification
retrieval
```

become possible.

---

# 56. The complete LLM picture

Now connect everything you've learned recently:

```text
                   TEXT
                    │
                    ▼
               TOKENIZATION
                    │
                    ▼
                TOKEN IDs
                    │
                    ▼
              EMBEDDING LOOKUP
                    │
                    ▼
            TOKEN EMBEDDINGS
                    │
                    ▼
          POSITIONAL INFORMATION
                    │
                    ▼
              TRANSFORMER
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
     Q / K / V             MLP
          │                   │
          ▼                   │
      ATTENTION               │
          │                   │
          └─────────┬─────────┘
                    ▼
             TRANSFORMER
                LAYERS
                    │
                    ▼
          CONTEXTUAL REPRESENTATION
                    │
                    ▼
               LM HEAD
                    │
                    ▼
          NEXT TOKEN PROBABILITY
                    │
                    ▼
              GENERATED TEXT
```

And RAG sits alongside it:

```text
                 YOUR DOCUMENTS
                       │
                       ▼
                    CHUNKS
                       │
                       ▼
                EMBEDDING MODEL
                       │
                       ▼
                  VECTOR DB
                       ▲
                       │
                  Query embedding
                       ▲
                       │
                    USER
                   QUESTION
                       │
                       ▼
                RETRIEVED CHUNKS
                       │
                       ▼
                     LLM
                       │
                       ▼
                    ANSWER
```

---

# 57. Embeddings vs Q/K/V — don't mix them up

Since you're learning these topics together:

| Concept | Main purpose |
|---|---|
| Token ID | Identify a token |
| Token embedding | Convert token ID into a vector |
| Positional information | Tell Transformer about position/order |
| Q | What information am I looking for? |
| K | What information do I contain/match? |
| V | What information should I provide? |
| Attention | Determine how information flows between tokens |
| Contextual representation | Meaning of a token after considering context |
| Sentence embedding | Represent an entire sentence as a vector |
| Document embedding | Represent a document/chunk as a vector |
| Vector database | Efficiently search embeddings |

The relationship is:

```text
TOKEN ID
   ↓
EMBEDDING
   ↓
X
   ↓
Q K V
   ↓
ATTENTION
   ↓
CONTEXTUAL REPRESENTATION
```

While RAG uses:

```text
DOCUMENT
   ↓
EMBEDDING MODEL
   ↓
VECTOR
   ↓
VECTOR DATABASE
   ↓
SIMILARITY SEARCH
```

---

# 58. What you should be able to do after learning this

If you've understood embeddings properly, you should now be able to explain:

### Basic

> What is an embedding?

A learned numerical vector representation of an object that allows useful relationships to be represented geometrically.

### LLM

> How does a token enter a Transformer?

```text
token ID
→ embedding lookup
→ vector
→ positional information
→ Transformer
```

### RAG

> Why are embeddings needed?

To convert documents and queries into vectors so that semantically relevant information can be retrieved using similarity search.

### Similarity

> How do we compare embeddings?

Commonly with cosine similarity, dot product, or Euclidean distance, depending on the system.

### Practical

> How do I build semantic search?

```text
documents
→ chunks
→ embeddings
→ vector database

query
→ embedding
→ similarity search
→ relevant chunks
```

---

# 59. Your practical learning exercise

Don't just read this. Do this once.

Install:

```bash
pip install sentence-transformers
```

Then create a Python file:

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Docker is a platform for building and running containers.",
    "Kubernetes orchestrates containerized applications.",
    "Python is a popular programming language.",
    "Git tracks changes in source code."
]

doc_embeddings = model.encode(
    documents,
    normalize_embeddings=True
)

while True:
    query = input("\nAsk something: ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    scores = util.dot_score(
        query_embedding,
        doc_embeddings
    )[0]

    results = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nMost relevant documents:")

    for document, score in results:
        print(f"{score:.4f} → {document}")
```

Try:

```text
What technology manages containers?
```

Then:

```text
How do I run applications inside containers?
```

Then:

```text
What programming language should I learn?
```

You will actually see the embedding space being used for retrieval.

---

# 60. The one sentence to remember

If you forget everything else, remember:

> **An embedding converts something such as a token, sentence, document, image, or audio clip into a learned vector so that relationships between those things can be represented and measured mathematically.**

And for your LLM journey, the two most important pipelines are:

```text
LLM:

Token
 ↓
Token ID
 ↓
Embedding
 ↓
Transformer
 ↓
Q/K/V Attention
 ↓
Contextual representation
 ↓
Prediction
```

and:

```text
RAG:

Document
 ↓
Chunk
 ↓
Embedding
 ↓
Vector DB

Query
 ↓
Embedding
 ↓
Similarity search
 ↓
Relevant chunks
 ↓
LLM
 ↓
Answer
```

Once **tokenization → embeddings → Q/K/V → attention → Transformer architecture** all click together, you have the core conceptual pipeline of a modern Transformer-based LLM.

