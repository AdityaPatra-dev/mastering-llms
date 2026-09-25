# Datasets Data Cleaning

I'll explain **Datasets + Data Cleaning** from the ground up, but with the goal that after reading this you can actually take a messy dataset, clean it, format it, tokenize it, split it, and prepare it for **SFT/LoRA/QLoRA training**.

# Datasets + Data Cleaning for LLMs

The most important idea to understand first is:

> **Fine-tuning is only as good as the data you train on.**

A powerful 7B model trained on poor data can become worse. A relatively small model trained on carefully constructed data can become very useful for a specific task.

Think of the entire process as:

```text
Raw data
   ↓
Collect
   ↓
Inspect
   ↓
Clean
   ↓
Remove duplicates
   ↓
Remove bad examples
   ↓
Standardize format
   ↓
Train/validation split
   ↓
Convert to model-friendly format
   ↓
Tokenize
   ↓
Dataset ready for SFT/LoRA/QLoRA
```

---

# 1. What exactly is a dataset?

A **dataset** is simply a structured collection of examples that your model can learn from.

For example, suppose you want to fine-tune a model to answer programming questions.

Your dataset could contain:

```text
Question:
What is a pointer in C?

Answer:
A pointer is a variable that stores the memory address
of another variable.
```

One such question-answer pair is one **training example**.

A dataset might contain:

```text
Example 1
Example 2
Example 3
...
Example 100,000
```

---

# 2. Dataset vs training data

These terms are often used interchangeably, but conceptually:

### Dataset

The complete collection of examples you have.

### Training set

The portion used to update model weights.

### Validation set

The portion used to monitor how well the model generalizes during training.

### Test set

A held-out portion used for final evaluation.

For example:

```text
100,000 examples
       │
       ├── 90,000 → training
       ├── 5,000  → validation
       └── 5,000  → test
```

A common split is:

```text
90% train
5% validation
5% test
```

But this isn't a law.

For a small dataset, you might use:

```text
80% train
10% validation
10% test
```

---

# 3. Why does data cleaning matter so much?

Imagine you're training a model with:

```text
User: What is recursion?

Assistant: Recursion is when a function calls itself.
```

That's good.

But your dataset also contains:

```text
User: What is recursion?

Assistant: I don't know lol.
```

And:

```text
User: What is recursion?

Assistant: Recursion is a programming language.
```

And:

```text
User: What is recursion?

Assistant: What is recursion?
```

And:

```text
User: What is recursion?

Assistant: [ERROR]
```

The model doesn't automatically know which examples are good.

It learns statistical patterns from the data.

So:

```text
Good data → useful patterns
Bad data → bad patterns
```

This is why **data quality often matters more than simply increasing dataset size**.

---

# 4. Types of datasets you'll encounter

For LLM work, you'll commonly encounter several formats.

## A. Plain text

Example:

```text
Python is a programming language.
It was created by Guido van Rossum.
```

This is useful for **continued pretraining / language modeling**.

You aren't explicitly telling the model:

```text
question → answer
```

You're simply giving it text.

---

# 5. Instruction dataset

This is extremely important for SFT.

Example:

```json
{
  "instruction": "Explain recursion.",
  "response": "Recursion is a technique where a function calls itself."
}
```

The model learns:

```text
instruction → response
```

---

# 6. Chat dataset

Modern instruction-tuned models often use conversational data.

For example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is recursion?"
    },
    {
      "role": "assistant",
      "content": "Recursion is a technique where a function calls itself."
    }
  ]
}
```

You can have multiple turns:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is recursion?"
    },
    {
      "role": "assistant",
      "content": "Recursion is when a function calls itself."
    },
    {
      "role": "user",
      "content": "Give me an example in C."
    },
    {
      "role": "assistant",
      "content": "Here is a simple example..."
    }
  ]
}
```

This is a very common format for modern SFT.

---

# 7. Preference datasets

These are used for things like preference optimization.

Example:

```json
{
  "prompt": "Explain pointers.",
  "chosen": "A pointer is a variable that stores a memory address.",
  "rejected": "A pointer is a type of loop."
}
```

You aren't just saying:

> This is the correct answer.

You're saying:

> Given these two answers, this one is preferred.

This becomes important later when you learn methods such as DPO.

---

# 8. What does "data cleaning" actually mean?

Data cleaning means:

> **Finding and fixing/removing data that could hurt the learning process.**

This can involve:

- missing values
- malformed examples
- duplicates
- irrelevant examples
- incorrect answers
- extremely long examples
- extremely short examples
- spam
- HTML
- broken Unicode
- corrupted text
- unwanted metadata
- inconsistent formatting
- personally identifiable information
- toxic or unsafe content depending on the task
- contradictory labels
- bad instruction/response pairs

---

# 9. The most important principle: cleaning depends on the goal

There is no universal definition of "clean."

Suppose you want to train a model for programming.

This:

```text
User:
How do I implement a linked list?

Assistant:
Use a struct containing data and a pointer to the next node.
```

is valuable.

But:

```text
User:
Write a poem about rain.

Assistant:
Rain falls softly...
```

may be irrelevant.

However, if you're building a general conversational model, that same example could be perfectly useful.

So:

```text
Good data ≠ universally good data

Good data = data appropriate for your objective
```

This is a very important concept.

---

# 10. Raw data is usually messy

Suppose you download a dataset containing:

```text
ID | Question | Answer | Website | Timestamp
```

You might find:

```text
001 | What is Python? | Python is... | example.com | ...
002 | What is Python? | Python is... | example.com | ...
003 |             | Python is... | example.com | ...
004 | What is Java? | Python is... | example.com | ...
005 | <html>... | ...
```

You can't blindly train on it.

You need a pipeline.

---

# 11. Step 1 — Understand your objective

Before touching the dataset, define:

### What should the model learn?

For example:

> I want a model that explains C programming concepts to beginners.

Then define:

### What should the model NOT learn?

For example:

- irrelevant programming languages
- random conversations
- spam
- incorrect explanations

Then define:

### What does a good example look like?

For example:

```text
Question → technically correct beginner-friendly explanation
```

This becomes your **data quality specification**.

---

# 12. Step 2 — Inspect the dataset

Never immediately start cleaning.

First inspect it.

Suppose you download:

```text
dataset.jsonl
```

You should determine:

```text
How many examples?
What fields exist?
Are fields missing?
How long are examples?
Are there duplicates?
What languages?
What topics?
What proportion is broken?
```

---

# 13. JSONL

You'll encounter JSONL constantly in ML.

JSONL means:

> **JSON Lines**

Instead of one huge JSON structure:

```json
[
  {"question": "What is C?", "answer": "A language."},
  {"question": "What is Python?", "answer": "A language."}
]
```

JSONL stores one JSON object per line:

```text
{"question":"What is C?","answer":"A programming language."}
{"question":"What is Python?","answer":"A programming language."}
{"question":"What is Java?","answer":"A programming language."}
```

This is convenient for large datasets.

---

# 14. Hugging Face Datasets

The Hugging Face ecosystem provides the `datasets` library.

Typically:

```bash
pip install datasets
```

Then:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="dataset.jsonl")
```

You can inspect it:

```python
print(dataset)
```

You might see:

```text
DatasetDict({
    train: Dataset({
        features: ['question', 'answer'],
        num_rows: 10000
    })
})
```

---

# 15. DatasetDict vs Dataset

This distinction is important.

A:

```text
Dataset
```

is basically a table.

For example:

```text
question              answer
----------------------------------------
What is C?             C is...
What is Python?        Python is...
```

A:

```text
DatasetDict
```

contains multiple splits:

```text
DatasetDict({
    train: ...
    validation: ...
    test: ...
})
```

Think:

```text
Dataset
    ↓
one table

DatasetDict
    ↓
collection of tables/splits
```

---

# 16. Inspect individual examples

You can do:

```python
print(dataset["train"][0])
```

Example:

```python
{
    "question": "What is recursion?",
    "answer": "Recursion occurs when a function calls itself."
}
```

You should inspect multiple random examples, not just the first one.

For example:

```python
import random

for i in random.sample(range(len(dataset["train"])), 5):
    print(dataset["train"][i])
```

This gives you a much better idea of the dataset.

---

# 17. Missing values

One of the first problems to detect is missing data.

Bad:

```json
{
  "question": "What is recursion?",
  "answer": ""
}
```

Or:

```json
{
  "question": "",
  "answer": "Recursion is..."
}
```

For instruction tuning, both are usually useless.

You can filter them.

Conceptually:

```python
def valid(example):
    return (
        example["question"] is not None
        and example["answer"] is not None
        and example["question"].strip() != ""
        and example["answer"].strip() != ""
    )
```

Then:

```python
dataset = dataset.filter(valid)
```

---

# 18. Why `.strip()`?

Suppose:

```text
"     What is Python?      "
```

Using:

```python
text.strip()
```

produces:

```text
"What is Python?"
```

It removes unnecessary whitespace at the beginning and end.

---

# 19. Duplicates

This is one of the biggest dataset problems.

Suppose you have:

```text
Example 1:
What is Python?
Python is a programming language.

Example 2:
What is Python?
Python is a programming language.
```

If this occurs thousands of times, your dataset isn't actually providing thousands of independent pieces of information.

The model will repeatedly see the same pattern.

---

# 20. Exact duplicates vs near duplicates

### Exact duplicate

Exactly the same text:

```text
A == A
```

### Near duplicate

Slightly different:

```text
What is Python?
Explain Python.

Python is a programming language.
```

They are semantically almost identical.

Removing exact duplicates is easy.

Near-duplicate detection is more advanced.

---

# 21. Exact deduplication

For example:

```python
dataset = dataset.drop_duplicates()
```

Depending on the library/version and workflow, you may instead create a unique key yourself or use pandas for tabular preprocessing.

Conceptually, you're doing:

```text
example → normalized text → hash → remove repeated hashes
```

---

# 22. Hashing

A hash converts data into a compact fingerprint.

For example:

```text
"What is Python?"
        ↓
hash
        ↓
8f31a...
```

Another identical string:

```text
"What is Python?"
        ↓
same hash
```

So you can detect duplicates efficiently.

You don't need to memorize hashing algorithms yet.

Just understand:

```text
same content
     ↓
same fingerprint
     ↓
duplicate detected
```

---

# 23. Near-duplicate detection

Suppose:

```text
Explain recursion.
```

and:

```text
Can you explain what recursion is?
```

Exact string matching won't identify them.

More advanced pipelines use:

- normalization
- n-gram similarity
- MinHash
- locality-sensitive hashing
- embeddings
- cosine similarity

For example:

```text
Text A → embedding vector
Text B → embedding vector
             ↓
      similarity score
```

If similarity is extremely high:

```text
0.98
```

they may be duplicates or near duplicates.

---

# 24. Don't blindly remove similar examples

This is important.

Suppose:

```text
What is a stack?
```

and:

```text
What is a queue?
```

Their wording may be similar.

But they teach different concepts.

Therefore:

> **Similarity does not automatically mean duplication.**

You need a threshold and, for important datasets, validation.

---

# 25. Bad answers

Suppose:

```text
Question:
What is a pointer?

Answer:
A pointer is a loop in C.
```

This is harmful.

If you're training a model on programming knowledge, incorrect answers can teach incorrect associations.

Therefore:

```text
Data cleaning
       +
Data validation
```

are both important.

---

# 26. Cleaning does not mean blindly deleting

This is a subtle but important distinction.

Suppose an answer contains a typo:

```text
Python is a programmng language.
```

You could fix:

```text
Python is a programming language.
```

But you need to be careful.

For large datasets, automatically modifying content can introduce errors.

Sometimes it's safer to:

```text
detect → flag → review
```

rather than:

```text
detect → automatically rewrite everything
```

---

# 27. HTML and markup

Web-scraped data often looks like:

```html
<h1>Python</h1>
<p>Python is a programming language.</p>
```

You may want:

```text
Python

Python is a programming language.
```

But don't blindly remove all markup.

For example, Markdown code blocks can be valuable:

```markdown
```python
print("Hello")
```
```

For a programming dataset, that code formatting is useful.

So cleaning should preserve meaningful structure.

---

# 28. Unicode problems

Datasets collected from the web may contain strange characters.

For example:

```text
PythonÂ is a programming language.
```

or inconsistent quotation marks:

```text
"hello"
“hello”
```

You may normalize text using Unicode normalization.

Python:

```python
import unicodedata

text = unicodedata.normalize("NFKC", text)
```

This can help make text more consistent.

But again, normalization should be appropriate to the data.

---

# 29. Language filtering

Suppose your project is specifically:

> English programming assistant.

But your dataset contains:

```text
English
Hindi
Japanese
Chinese
Spanish
Russian
```

You may need language identification.

You can classify:

```text
example → language detector → English / not English
```

Then filter appropriately.

However, multilingual data can be useful if you intentionally want a multilingual model.

---

# 30. Length filtering

Some examples might be:

```text
Question: Hi
Answer: Hello
```

while others are:

```text
Question:
Explain the complete architecture of a modern distributed
database including...
[50,000 tokens]
```

Extreme examples can cause problems.

You might establish minimum and maximum lengths.

For example:

```python
def valid_length(example):
    return (
        10 <= len(example["answer"]) <= 10000
    )
```

But **character length is not the same as token length**.

That's important.

---

# 31. Characters vs tokens

Suppose:

```text
Hello world
```

might be a few tokens.

But:

```text
supercalifragilisticexpialidocious
```

can be tokenized differently depending on the tokenizer.

The model ultimately operates on:

```text
tokens
```

not characters.

Therefore, for serious LLM preprocessing, you eventually want to inspect **token lengths**.

---

# 32. Why maximum sequence length matters

Suppose your model supports:

```text
8192 tokens
```

but you have examples with:

```text
20,000 tokens
```

You can't simply feed the entire example into the model.

You need to decide whether to:

- truncate it
- split it
- chunk it
- remove it
- use a model/configuration with a larger context length

Blind truncation can be dangerous.

Imagine:

```text
Question
   ↓
long explanation
   ↓
important conclusion
```

If you truncate the end, you may remove the actual answer.

---

# 33. Instruction-response consistency

Suppose your dataset has:

```json
{
  "instruction": "Explain Python.",
  "response": "Java is an object-oriented language."
}
```

Structurally valid.

Semantically wrong.

This is harder to detect automatically.

You may need:

- rule-based checks
- human review
- another model as a quality filter
- domain-specific validators

---

# 34. Data contamination

This becomes extremely important when evaluating models.

Suppose your test set contains:

```text
Question:
What is a binary search tree?
```

And the exact same example appears in your training dataset.

Then the model may simply memorize it.

Your evaluation becomes misleading.

Therefore:

```text
TRAIN
  ↓
should not overlap
  ↓
TEST
```

---

# 35. Train/validation/test leakage

Imagine:

```text
Training:
"What is recursion?"
"Recursion is..."

Test:
"What is recursion?"
"Recursion is..."
```

Your test score might look excellent.

But the model hasn't necessarily learned the underlying concept.

It may have memorized the answer.

This is called **data leakage**.

---

# 36. Splitting the dataset

After cleaning:

```text
Clean dataset
       ↓
split
       ↓
┌───────────────┐
│ train         │
│ validation    │
│ test          │
└───────────────┘
```

With Hugging Face:

```python
split = dataset.train_test_split(test_size=0.1)
```

This gives:

```text
train
test
```

You can further split training into train/validation.

---

# 37. Random splitting isn't always enough

Imagine your dataset contains:

```text
1000 examples about C
1000 examples about Python
1000 examples about Java
```

A random split may be okay.

But suppose it contains documents:

```text
Document A
 ├── chunk 1
 ├── chunk 2
 ├── chunk 3
 └── chunk 4
```

If you randomly split chunks:

```text
train → chunk 1
train → chunk 2
test  → chunk 3
```

the model has already seen almost the same document during training.

That's leakage.

Instead, split by **document/source**, then create chunks.

---

# 38. Source-aware splitting

Better:

```text
Documents
   ↓
split documents
   ↓
train documents
test documents
   ↓
chunk each separately
```

This produces a more honest evaluation.

---

# 39. Dataset formatting

Suppose your raw data looks like:

```json
{
  "question": "What is a linked list?",
  "answer": "A linked list is..."
}
```

But the model expects:

```text
<user>
What is a linked list?

<assistant>
A linked list is...
```

You need to transform the data.

For chat models, this is often done through a chat template.

Conceptually:

```text
Raw fields
    ↓
message structure
    ↓
chat template
    ↓
model-ready text
```

---

# 40. Chat templates

Different models can expect different special tokens.

For example, internally a conversation may look like:

```text
<|system|>
You are a helpful assistant.
<|user|>
Explain recursion.
<|assistant|>
Recursion is...
```

Another model may use completely different tokens.

Therefore:

> **Do not manually invent special tokens if the model already provides a chat template.**

With Hugging Face tokenizers, you will often use:

```python
tokenizer.apply_chat_template(...)
```

The exact syntax depends on the model.

---

# 41. Why formatting matters

Suppose the base model was trained to understand:

```text
user → assistant
```

but you fine-tune it with:

```text
Question:
...
Answer:
...
```

It may still work.

But using the model's intended chat format can make training more consistent with how the model was originally instruction-tuned.

---

# 42. Dataset formatting example

Imagine:

```python
example = {
    "question": "What is recursion?",
    "answer": "Recursion is when a function calls itself."
}
```

You might transform it into:

```python
{
    "messages": [
        {
            "role": "user",
            "content": "What is recursion?"
        },
        {
            "role": "assistant",
            "content": "Recursion is when a function calls itself."
        }
    ]
}
```

Now it is much closer to a conversational SFT dataset.

---

# 43. Multi-turn conversations

Suppose you're building a coding tutor.

A good example could be:

```text
User:
What is a linked list?

Assistant:
A linked list is...

User:
Why use it instead of an array?

Assistant:
A linked list allows...

User:
Show me C code.

Assistant:
Here is a simple implementation...
```

This teaches the model how conversation works.

---

# 44. System messages

You can also have:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a C programming tutor."
    },
    {
      "role": "user",
      "content": "Explain pointers."
    },
    {
      "role": "assistant",
      "content": "A pointer is..."
    }
  ]
}
```

This can teach behavior/style.

But don't overuse system messages if they don't represent the behavior you actually want.

---

# 45. Data diversity

A dataset should ideally cover the different situations you expect the model to handle.

Suppose you're training a C tutor.

Bad dataset:

```text
What is a pointer?
What is a pointer?
Explain pointers.
Define pointers.
What is pointer?
```

Better:

```text
What is a pointer?
Explain pointer arithmetic.
Why use pointers?
What is a NULL pointer?
What is a dangling pointer?
How does malloc work?
Write a linked list using pointers.
Find the bug in this pointer code.
Explain double pointers.
```

Now the model sees a broader distribution of tasks.

---

# 46. Data balance

Suppose:

```text
80% → pointers
10% → arrays
5%  → linked lists
5%  → trees
```

If your goal is a general C tutor, the model may disproportionately learn pointers.

This is a **data distribution problem**.

You might want something more balanced.

Not necessarily perfectly equal, but aligned with your objective.

---

# 47. Quality beats quantity

Imagine two datasets.

### Dataset A

```text
1,000,000 mediocre examples
```

### Dataset B

```text
50,000 carefully curated examples
```

Dataset B can potentially be much more useful for a specialized SFT task.

This is why modern LLM pipelines spend significant effort on:

```text
data filtering
data deduplication
data quality
data composition
```

---

# 48. Synthetic data

You don't always need humans to write every example.

You can use an existing LLM to generate training examples.

For example:

```text
Topic:
C pointers

        ↓

LLM generates

Question:
Explain pointer arithmetic.

Answer:
...
```

Then you filter the generated data.

This is called **synthetic data**.

---

# 49. Synthetic data danger

LLMs can generate:

```text
confidently incorrect answers
```

So:

```text
LLM-generated data
        ↓
quality validation
        ↓
accepted examples
```

not:

```text
LLM-generated data
        ↓
automatically train
```

---

# 50. Data cleaning pipeline

A practical pipeline might look like:

```text
                RAW DATA
                    │
                    ▼
             Schema validation
                    │
                    ▼
              Remove missing
                    │
                    ▼
             Normalize text
                    │
                    ▼
             Remove duplicates
                    │
                    ▼
          Remove irrelevant data
                    │
                    ▼
         Check quality / correctness
                    │
                    ▼
           Length/token filtering
                    │
                    ▼
          Safety/privacy filtering
                    │
                    ▼
             Format examples
                    │
                    ▼
          Train/validation/test split
                    │
                    ▼
               Tokenization
                    │
                    ▼
             FINAL DATASET
```

That's the mental model I want you to remember.

---

# 51. Schema validation

A schema describes what an example should contain.

For example:

```text
messages → list
messages[].role → string
messages[].content → string
```

Valid:

```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
  ]
}
```

Invalid:

```json
{
  "messages": "Hello"
}
```

You can automatically reject malformed records.

---

# 52. A practical cleaning function

Imagine your raw dataset:

```python
def clean_example(example):

    question = example["question"]
    answer = example["answer"]

    if question is None or answer is None:
        return None

    question = question.strip()
    answer = answer.strip()

    if not question or not answer:
        return None

    return {
        "question": question,
        "answer": answer
    }
```

Then you can apply it to every example.

In real pipelines, you'd usually design the filtering/mapping operations so the resulting dataset remains valid rather than literally returning `None` without handling it.

---

# 53. Filtering vs mapping

This distinction is extremely useful.

### `filter`

Used when asking:

> Should I keep this example?

```text
example
   ↓
TRUE → keep
FALSE → remove
```

Example:

```python
dataset.filter(lambda x: len(x["answer"]) > 20)
```

---

### `map`

Used when asking:

> How should I transform this example?

```text
old example
     ↓
transformation
     ↓
new example
```

Example:

```python
dataset.map(format_example)
```

So remember:

```text
filter → keep/remove

map → transform
```

---

# 54. Cleaning vs preprocessing vs tokenization

These are related but different.

### Cleaning

Fix/remove bad data.

```text
bad → good
```

### Preprocessing

Transform data into the structure needed by training.

```text
raw structure → training structure
```

### Tokenization

Convert text into token IDs.

```text
text → tokens → integers
```

For example:

```text
"Hello world"
       ↓
["Hello", " world"]
       ↓
[15496, 995]
```

The exact token IDs depend on the tokenizer.

---

# 55. Tokenization happens later

A typical workflow:

```text
Raw dataset
     ↓
Cleaning
     ↓
Formatting
     ↓
Splitting
     ↓
Tokenization
     ↓
Training
```

Don't confuse:

```text
cleaning
```

with:

```text
tokenization
```

---

# 56. Tokenizer compatibility

This is crucial.

If you're fine-tuning:

```text
Qwen model
```

use its tokenizer.

If you're fine-tuning:

```text
Llama model
```

use its tokenizer.

Don't randomly use another model's tokenizer.

The model's embeddings and vocabulary are tied to its tokenizer.

---

# 57. Token length analysis

After choosing your model/tokenizer, inspect:

```text
minimum tokens
median tokens
average tokens
95th percentile
maximum tokens
```

For example:

```text
Min:       15
Median:    240
Mean:      310
95th:      900
Max:       15,000
```

This tells you how your dataset interacts with your context length.

---

# 58. Why percentiles matter

Suppose:

```text
99% examples < 1000 tokens
1% examples = 100,000 tokens
```

The mean might become misleading.

The 95th or 99th percentile tells you more about the normal distribution.

For example:

```text
P50 = 220 tokens
P90 = 600
P95 = 850
P99 = 1800
```

Now you know most examples are relatively short.

---

# 59. Packing

Suppose your model supports:

```text
2048 tokens
```

and you have:

```text
Example A = 300 tokens
Example B = 500 tokens
Example C = 400 tokens
```

You may be able to pack multiple short examples into a training sequence:

```text
300 + 500 + 400 = 1200
```

instead of wasting the remaining context.

This is called **packing**.

It's a training optimization, not data cleaning itself.

---

# 60. Privacy and PII

Web datasets may contain:

```text
names
emails
phone numbers
addresses
API keys
passwords
tokens
```

You generally don't want to blindly train on such information.

Especially:

```text
sk_test_...
API_KEY=...
password=...
```

These can be dangerous to expose or memorize.

So data pipelines may include:

```text
PII detection
secret detection
redaction
removal
```

For example:

```text
my email is abc@example.com
```

could become:

```text
my email is [EMAIL]
```

depending on the use case.

---

# 61. Copyright and licensing

When building public datasets, another major consideration is:

> **Are you legally allowed to redistribute/use this data?**

You should check:

- dataset license
- source license
- redistribution restrictions
- model/data usage terms
- attribution requirements

This becomes especially important if you're planning to upload a dataset to Hugging Face.

---

# 62. Dataset cards

When publishing a dataset on Hugging Face, a **dataset card** explains things such as:

```text
What is this dataset?
Where did it come from?
How was it collected?
How was it cleaned?
What license applies?
What are limitations?
What biases might exist?
What should it be used for?
```

This is part of responsible dataset publishing.

---

# 63. Dataset versioning

Suppose you have:

```text
dataset_v1
dataset_v2
dataset_v3
```

You should know what changed.

For example:

```text
v1 → 100k examples
v2 → duplicates removed
v3 → incorrect examples removed
```

This makes experiments reproducible.

---

# 64. Reproducibility

Imagine you train:

```text
Model A
```

on:

```text
dataset_v2
```

Then later you can't remember:

- which examples were removed
- which filtering rules were used
- which tokenizer
- which split seed
- which formatting function

You won't be able to reproduce the result.

So keep your pipeline as code.

Example:

```text
data/
  raw/
  cleaned/
  processed/

scripts/
  clean.py
  deduplicate.py
  format.py
  split.py
```

---

# 65. A practical project structure

For your own Hugging Face project, I'd recommend something like:

```text
my-llm-dataset/
│
├── data/
│   ├── raw/
│   │   └── raw.jsonl
│   │
│   ├── cleaned/
│   │   └── cleaned.jsonl
│   │
│   └── processed/
│       └── train.jsonl
│       └── validation.jsonl
│       └── test.jsonl
│
├── scripts/
│   ├── inspect.py
│   ├── clean.py
│   ├── deduplicate.py
│   ├── format.py
│   └── analyze_tokens.py
│
├── README.md
└── requirements.txt
```

This is already approaching a real-world data pipeline.

---

# 66. Your first practical dataset

Let's say you want to build:

> **C Programming Tutor Dataset**

You could start with:

```json
{"question":"What is a pointer in C?","answer":"A pointer is a variable that stores the memory address of another object."}
{"question":"What is malloc?","answer":"malloc dynamically allocates a block of memory and returns a pointer to it."}
{"question":"What is recursion?","answer":"Recursion is a technique where a function directly or indirectly calls itself."}
```

Then convert:

```text
question + answer
```

into:

```text
messages
```

---

# 67. Cleaning this dataset

You'd check:

### Missing fields

```text
question = NULL
```

Remove.

### Empty strings

```text
answer = ""
```

Remove.

### Duplicates

```text
same Q/A repeated
```

Remove.

### Incorrect answers

```text
malloc allocates CPU cores
```

Remove/fix.

### Irrelevant content

```text
"Who won yesterday's cricket match?"
```

Remove if your objective is C programming.

### Extremely long content

Inspect and decide whether to split/remove.

---

# 68. Then format it

Convert:

```json
{
  "question": "What is malloc?",
  "answer": "malloc dynamically allocates memory."
}
```

into:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is malloc?"
    },
    {
      "role": "assistant",
      "content": "malloc dynamically allocates memory."
    }
  ]
}
```

---

# 69. Then split it

For example:

```text
10,000 cleaned examples

        ↓

8,000 train
1,000 validation
1,000 test
```

But ideally perform your leakage/duplicate checks with the split strategy in mind.

---

# 70. Then tokenize

Using the target model's tokenizer:

```text
"What is malloc?"
       ↓
tokenizer
       ↓
[What, is, mal, loc, ?]
       ↓
token IDs
```

The actual tokens depend on the tokenizer.

---

# 71. Then inspect token lengths

You might discover:

```text
Most examples: 100–500 tokens
Some: 5,000 tokens
Some: 20,000 tokens
```

Now you can decide:

```text
short → keep
normal → keep
very long → chunk/review
```

---

# 72. Then you're ready for SFT

At this point:

```text
Dataset
   ↓
clean
   ↓
deduplicate
   ↓
validate
   ↓
format
   ↓
split
   ↓
tokenize
   ↓
SFT
   ↓
LoRA / QLoRA
```

This connects directly to what you've been learning:

```text
Fine-tuning / SFT
        ↑
        │
Datasets + cleaning
        ↑
        │
Tokenization
        ↑
        │
Transformer
```

---

# 73. A complete miniature pipeline

Conceptually:

```python
from datasets import load_dataset

# 1. Load
dataset = load_dataset(
    "json",
    data_files="raw.jsonl"
)

# 2. Remove missing/empty examples
def valid(example):
    question = example["question"]
    answer = example["answer"]

    return (
        question is not None
        and answer is not None
        and question.strip() != ""
        and answer.strip() != ""
    )

dataset = dataset.filter(valid)

# 3. Normalize
def normalize(example):
    return {
        "question": example["question"].strip(),
        "answer": example["answer"].strip()
    }

dataset = dataset.map(normalize)

# 4. Convert to chat format
def make_messages(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": example["question"]
            },
            {
                "role": "assistant",
                "content": example["answer"]
            }
        ]
    }

dataset = dataset.map(make_messages)

# 5. Split
split = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42
)

print(split)
```

This is not yet a production-grade cleaning pipeline, but it demonstrates the fundamental workflow.

---

# 74. Why `seed=42`?

When randomly splitting data:

```python
seed=42
```

makes the random operation reproducible.

Without a fixed seed:

```text
Run 1 → example X goes to train
Run 2 → example X goes to test
```

With a fixed seed:

```text
Run 1 → same split
Run 2 → same split
```

The number `42` isn't magical. Any fixed seed can work.

---

# 75. Dataset quality checklist

Before training, ask:

### Structure

- Does every example have the required fields?
- Are there malformed records?

### Completeness

- Are questions missing?
- Are answers missing?
- Are conversations incomplete?

### Correctness

- Are answers factually correct?
- Are labels correct?

### Relevance

- Does the example match the task?

### Duplication

- Are there exact duplicates?
- Are there near duplicates?

### Length

- Are examples too short?
- Are examples too long?
- What is the token distribution?

### Diversity

- Are there enough different tasks/topics/styles?

### Balance

- Is one category dominating?

### Leakage

- Does test data appear in training?
- Are chunks from the same document split across train/test?

### Privacy

- Are there emails?
- API keys?
- passwords?
- personal information?

### Licensing

- Are you legally allowed to use and redistribute it?

### Formatting

- Does it match the target model's expected conversational format?

---

# 76. The biggest beginner mistake

A beginner often thinks:

```text
Download dataset
       ↓
train model
```

A better mental model is:

```text
                    DATA
                     │
             ┌───────┴────────┐
             │                │
         Quantity          Quality
             │                │
             └───────┬────────┘
                     ↓
                 Cleaning
                     ↓
              Deduplication
                     ↓
                Validation
                     ↓
                Formatting
                     ↓
                  Splits
                     ↓
                Tokenization
                     ↓
                  Training
```

---

# 77. Another important concept: distribution

Your dataset teaches the model:

> **What kinds of inputs and outputs it should expect.**

Suppose you train a coding assistant exclusively on:

```text
"Write Python code."
```

Then don't be surprised if it becomes very specialized around Python.

Your dataset distribution determines what behaviors the model gets exposed to.

For example:

```text
40% explanations
20% debugging
15% code generation
10% code review
10% conceptual questions
5% multi-turn conversations
```

could produce a different behavior from:

```text
90% code generation
10% everything else
```

So dataset composition is part of **model behavior engineering**.

---

# 78. Data mixture

You can combine multiple datasets.

For example:

```text
C dataset
+
Python dataset
+
Linux dataset
+
Git dataset
+
Docker dataset
```

to create:

```text
Developer Assistant Dataset
```

But you need to consider proportions.

If:

```text
Python = 500,000
C = 5,000
Docker = 2,000
```

Python will dominate unless you deliberately rebalance or sample.

---

# 79. Curriculum vs mixture

Sometimes you may intentionally structure what the model sees.

For example:

```text
basic concepts
      ↓
intermediate concepts
      ↓
advanced problems
```

This resembles a curriculum.

But for ordinary SFT, you generally don't need to manually create a perfect curriculum. Good diverse data and appropriate training setup matter more.

---

# 80. Data augmentation

Suppose you have:

```text
Explain recursion.
```

You might generate variations:

```text
What is recursion?

Explain recursion to a beginner.

How does recursion work?

Why would a programmer use recursion?

Explain recursion with an example.
```

This increases variation.

But be careful:

```text
augmentation ≠ automatically better
```

Poorly generated variations can add noise.

---

# 81. Human review

For high-quality datasets, humans can review examples.

For example:

```text
Example
   ↓
Automatic filters
   ↓
95% accepted
5% flagged
   ↓
Human review
   ↓
final dataset
```

You don't necessarily need humans to inspect millions of examples.

Instead, use automation to find suspicious examples and humans to review difficult cases.

---

# 82. Automated quality scoring

You can assign:

```text
quality_score = 0.0 → 1.0
```

based on things such as:

```text
correctness
relevance
completeness
clarity
```

Then:

```text
score < 0.3 → remove
0.3–0.7 → review
> 0.7 → keep
```

But be careful: an LLM-generated quality score is not ground truth.

Use such scores as **filters or prioritization**, not unquestionable truth.

---

# 83. Data cleaning for your local LLM goal

For your specific goal of eventually creating **coding/DevOps/web-development RAG + fine-tuned models**, I'd separate your data into categories.

For example:

```text
coding_sft/
│
├── C/
├── C++/
├── Python/
├── Java/
├── Git/
├── Linux/
├── Docker/
├── Kubernetes/
└── web/
```

And separately:

```text
RAG/
│
├── documentation/
├── college/
├── project_docs/
└── references/
```

This is an important distinction.

---

# 84. SFT dataset vs RAG data

Don't confuse these.

### SFT

You modify model behavior/knowledge through training.

```text
dataset
   ↓
training
   ↓
model weights change
```

### RAG

You keep information outside the model.

```text
documents
   ↓
chunks
   ↓
embeddings
   ↓
vector database
   ↓
retrieval
   ↓
LLM
```

For example, your Docker documentation doesn't necessarily need to be baked into the model through fine-tuning.

You could use:

```text
Docker docs → RAG
```

while using SFT for:

```text
desired coding assistant behavior
```

That's a very useful architecture.

---

# 85. SFT data should teach behavior, not necessarily store every fact

This is a subtle but powerful idea.

Suppose you want:

> A model that answers Docker questions using retrieved documentation.

You might use SFT to teach:

```text
how to reason
how to structure answers
how to interpret retrieved context
how to write commands
how to admit uncertainty
```

And RAG to provide:

```text
current Docker documentation
```

This prevents you from trying to solve everything through fine-tuning.

---

# 86. Your practical learning progression

Given what you're learning now, I would follow:

```text
1. Dataset fundamentals
        ↓
2. JSON / JSONL
        ↓
3. Hugging Face Datasets
        ↓
4. Filtering
        ↓
5. Mapping
        ↓
6. Deduplication
        ↓
7. Dataset splitting
        ↓
8. Token-length analysis
        ↓
9. Chat formatting
        ↓
10. Tokenization
        ↓
11. SFT
        ↓
12. LoRA
        ↓
13. QLoRA
```

You've already been learning the later parts, so **dataset preparation is the missing practical bridge**.

---

# 87. The complete picture of your LLM training pipeline

Put everything you've learned recently together:

```text
                    RAW DATA
                       │
                       ▼
              DATA COLLECTION
                       │
                       ▼
              DATA CLEANING
          ┌────────────┼────────────┐
          ↓            ↓            ↓
      filtering     dedup       validation
          └────────────┼────────────┘
                       ↓
                DATA FORMATTING
                       ↓
               TRAIN/VAL/TEST
                       ↓
                 TOKENIZATION
                       ↓
                 ┌───────────┐
                 │    SFT    │
                 └─────┬─────┘
                       ↓
                    LoRA
                       ↓
                   QLoRA
                       ↓
              Fine-tuned model
                       ↓
                  Evaluation
                       ↓
                  Deployment
```

And underneath all of this:

```text
Transformer
   ↓
Tokenization
   ↓
Embeddings
   ↓
Attention Q/K/V
   ↓
Causal LM
   ↓
SFT
   ↓
LoRA
   ↓
QLoRA
   ↓
Dataset
```

These aren't separate random topics. **They're pieces of one pipeline.**

---

# 88. What you should be able to do after learning this

If you've genuinely understood this chapter, you should now be able to take something like:

```text
raw_data.jsonl
```

and reason through:

```text
Is the schema correct?
        ↓
Are records missing?
        ↓
Are there duplicates?
        ↓
Are answers correct?
        ↓
Is the data relevant?
        ↓
Is the distribution appropriate?
        ↓
Are there privacy problems?
        ↓
Is there train/test leakage?
        ↓
What format does my model expect?
        ↓
How long are the examples in tokens?
        ↓
How should I split them?
        ↓
How should I tokenize them?
        ↓
Is this dataset ready for SFT?
```

That is the **practical skill** you want.

---

# 89. The 10 things I want you to remember

If you forget everything else, remember these:

### 1. Dataset ≠ just a pile of text

It represents the examples and distribution from which your model learns.

### 2. Quality matters enormously

Bad data can teach bad behavior.

### 3. Cleaning is task-dependent

"Good data" depends on what you're training the model to do.

### 4. `filter` removes examples

```text
keep/remove
```

### 5. `map` transforms examples

```text
old → new
```

### 6. Duplicates can distort training

Especially massive repeated examples.

### 7. Train/test leakage ruins evaluation

Never allow evaluation examples to effectively appear in training.

### 8. Token length matters

Models operate on tokens and have context limits.

### 9. Formatting matters

For chat models, use the appropriate message structure/chat template.

### 10. Dataset preparation is part of model engineering

It's not boring preparation before the "real ML."

**The dataset is one of the core components of the model.**

---

## Your first practical project

Now that you understand the concepts, don't immediately jump to a 100-million-example dataset.

Build this:

```text
C Programming SFT Dataset
```

Start with **100–500 examples**.

Create:

```text
question
answer
```

Then write a Python pipeline that:

```text
raw JSONL
   ↓
validate
   ↓
clean whitespace
   ↓
remove empty examples
   ↓
remove duplicates
   ↓
inspect examples
   ↓
convert → messages
   ↓
train/validation/test split
   ↓
load Qwen/Llama tokenizer
   ↓
calculate token-length statistics
   ↓
save final dataset
```

Then you'll have the exact foundation needed before moving into your **SFT → LoRA → QLoRA practical training project**.

And one particularly important next step for you is **tokenization in practice**: you've learned what tokenization is conceptually, but now you should learn how to take this cleaned dataset and use a Hugging Face tokenizer to produce the actual `input_ids`, `attention_mask`, labels, padding, truncation, and chat-template output that SFT training consumes.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)