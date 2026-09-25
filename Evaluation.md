# Evaluation Practically

I'll explain **Evaluation** specifically in that context—from the basic idea all the way to evaluating a fine-tuned model practically.

# Evaluation of ML/LLM Models

## 1. What exactly is "evaluation"?

Suppose you fine-tune a model to do this:

> **Input:** "Explain recursion in C."  
> **Expected output:** A correct explanation of recursion.

After training, you need to answer:

> **"Did my model actually become better?"**

That's **evaluation**.

Training tells the model **how to improve its parameters**.

Evaluation tells **you how well it performs on data it wasn't trained on**.

The basic ML workflow is:

```text
Dataset
   ↓
Clean
   ↓
Train / Fine-tune
   ↓
Model
   ↓
Evaluation
   ↓
Metrics + Error Analysis
   ↓
Improve dataset/model/training
   ↓
Evaluate again
```

The most important concept is:

> **Never judge a model only on the data it trained on.**

A model can memorize training examples and still perform badly on new examples.

---

# 2. Train, validation, and test sets

This is the foundation of evaluation.

Suppose you have:

```text
10,000 examples
```

You might divide them into:

```text
Training       8,000
Validation     1,000
Test           1,000
```

### Training set

Used to actually train the model.

```text
Training examples
       ↓
    model
       ↓
update weights
```

The model sees these repeatedly.

---

### Validation set

Used during development.

You don't train directly on it.

You use it to answer questions like:

- Which checkpoint is best?
- Is the model overfitting?
- Should I train for another epoch?
- Is learning rate too high?
- Is one model configuration better than another?

Example:

```text
Epoch 1 → validation loss = 2.8
Epoch 2 → validation loss = 2.1
Epoch 3 → validation loss = 1.9
Epoch 4 → validation loss = 2.0
```

This suggests that performance stopped improving around epoch 3.

---

### Test set

The test set is your **final exam**.

You ideally don't use it while developing the model.

After you've finalized:

- dataset
- preprocessing
- LoRA configuration
- learning rate
- epochs
- model
- prompt format

you evaluate once on the test set.

```text
Training set
     ↓
Training

Validation set
     ↓
Model selection / tuning

Test set
     ↓
Final evaluation
```

This distinction is extremely important.

---

# 3. Why can't we just look at loss?

Because **loss is not necessarily the same thing as useful performance**.

Imagine two models:

### Model A

```text
Loss = 1.8
```

### Model B

```text
Loss = 1.5
```

You might think B is better.

But suppose:

```text
Model A:
- follows instructions well
- gives concise answers
- rarely hallucinates

Model B:
- lower loss
- often produces irrelevant answers
- sometimes ignores instructions
```

Then loss alone doesn't tell the whole story.

Evaluation should answer:

> **Does the model perform the task we actually care about?**

---

# 4. Evaluation depends on the task

There isn't one universal metric for LLMs.

Different tasks require different evaluation methods.

For example:

| Task | Useful evaluation |
|---|---|
| Classification | Accuracy, Precision, Recall, F1 |
| Sentiment analysis | Accuracy/F1 |
| Translation | BLEU, chrF, COMET |
| Summarization | ROUGE, BERTScore + human evaluation |
| Question answering | Exact Match, F1 |
| Code generation | Pass@k, unit tests |
| Chatbot | Human evaluation, LLM-as-judge, task-specific metrics |
| Instruction tuning | Task success, human/LLM evaluation |
| Language modeling | Perplexity |
| Retrieval | Recall@k, Precision@k, MRR, nDCG |

So before choosing a metric, ask:

> **What does "good" mean for my model?**

---

# 5. Training loss vs evaluation loss

This is one of the most important concepts.

During training:

```text
training examples
      ↓
    model
      ↓
 prediction
      ↓
 calculate loss
      ↓
 update weights
```

During evaluation:

```text
unseen examples
      ↓
    model
      ↓
 prediction
      ↓
 calculate loss
      ↓
NO weight update
```

Notice:

```text
Training → weights change
Evaluation → weights DON'T change
```

That's a fundamental difference.

---

# 6. What is overfitting?

Suppose you train a model.

You observe:

```text
             Training loss    Validation loss

Epoch 1          2.5               2.7
Epoch 2          1.9               2.1
Epoch 3          1.4               1.8
Epoch 4          1.0               1.9
Epoch 5          0.7               2.2
```

Training loss keeps improving.

But validation loss starts getting worse.

That's a classic sign of:

> **Overfitting**

The model is becoming increasingly good at the training data but increasingly poor at generalizing.

Visually:

```text
Loss
 ^
 |\
 | \
 |  \       training
 |   \
 |    \
 |     \
 |      \________
 |
 |       \___/\__ validation
 |
 +--------------------> epochs
```

The ideal checkpoint might be around epoch 3.

---

# 7. Generalization

This is the real goal.

Suppose training examples are:

```text
What is recursion?
What is a pointer?
What is malloc?
```

If your model performs well on completely new questions:

```text
Why can recursion cause stack overflow?
When should you use calloc instead of malloc?
```

then it has **generalized**.

So:

> **Generalization = performing well on previously unseen data.**

Good evaluation measures generalization.

---

# 8. Classification evaluation

Let's first understand traditional ML metrics because these concepts are useful everywhere.

Suppose you're building a spam detector.

```text
Prediction:
Spam / Not Spam
```

There are four possibilities.

### True Positive

Actual:

```text
Spam
```

Model:

```text
Spam
```

Correct.

### True Negative

Actual:

```text
Not spam
```

Model:

```text
Not spam
```

Correct.

### False Positive

Actual:

```text
Not spam
```

Model:

```text
Spam
```

Wrong.

### False Negative

Actual:

```text
Spam
```

Model:

```text
Not spam
```

Wrong.

This gives the famous:

```text
                 Actual
              Spam   Not Spam
Prediction
Spam           TP       FP
Not Spam       FN       TN
```

---

# 9. Accuracy

Accuracy asks:

> "What fraction of predictions were correct?"

Formula:

$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

Example:

```text
100 predictions

90 correct
10 incorrect
```

Accuracy:

```text
90 / 100 = 90%
```

Easy.

But accuracy can be misleading.

---

# 10. Why accuracy can be dangerous

Suppose you have:

```text
10,000 emails

9,900 = normal
100 = spam
```

A terrible model predicts:

```text
EVERYTHING = normal
```

It gets:

```text
9,900 / 10,000 = 99%
```

Accuracy = **99%**

Sounds fantastic.

But it detects:

```text
0 spam messages
```

So accuracy isn't enough when classes are imbalanced.

---

# 11. Precision

Precision asks:

> **Of everything the model predicted as positive, how much was actually positive?**

$$
Precision = \frac{TP}{TP+FP}
$$

Example:

Model says:

```text
100 messages = spam
```

Actually:

```text
80 = spam
20 = normal
```

Therefore:

```text
Precision = 80 / 100
          = 80%
```

High precision means:

> When the model says "positive", it is usually correct.

---

# 12. Recall

Recall asks:

> **Of all the actual positive examples, how many did the model find?**

$$
Recall = \frac{TP}{TP+FN}
$$

Suppose:

```text
100 actual spam messages
```

Model detects:

```text
80
```

Then:

```text
Recall = 80 / 100
       = 80%
```

High recall means:

> The model doesn't miss many positive examples.

---

# 13. Precision vs recall

Imagine a security system detecting malicious files.

### High precision

The model is conservative.

It only calls something malicious when it's very confident.

Result:

```text
few false alarms
```

But it may miss some malicious files.

---

### High recall

The model is aggressive.

It catches almost everything suspicious.

But:

```text
more false alarms
```

So there is often a trade-off.

---

# 14. F1 score

F1 combines precision and recall.

$$
F1 = 2\frac{Precision \times Recall}
{Precision + Recall}
$$

Suppose:

```text
Precision = 0.8
Recall = 0.6
```

Then:

$$
F1 \approx 0.686
$$

F1 is useful when both precision and recall matter.

---

# 15. Perplexity

Now we're getting directly into LLM evaluation.

For language models, you will frequently encounter:

> **Perplexity (PPL)**

It measures how well a language model predicts tokens.

Very roughly:

> **How surprised is the model by the actual text?**

Suppose:

```text
The cat sat on the ___
```

A model might assign:

```text
mat       0.70
floor     0.15
chair     0.05
table     0.03
...
```

If the actual word is:

```text
mat
```

the model was not very surprised.

If the actual word was:

```text
spaceship
```

and the model assigned it probability:

```text
0.0001
```

the model was extremely surprised.

---

# 16. Perplexity mathematically

For a sequence of tokens:

$$
x_1,x_2,...,x_N
$$

the model calculates:

$$
P(x_1,x_2,...,x_N)
$$

The average negative log likelihood is related to cross-entropy.

Perplexity is:

$$
PPL = e^{Loss}
$$

when using natural logarithms.

So approximately:

```text
Loss ↓
Perplexity ↓
```

Generally, lower perplexity means the model predicts the evaluation text better.

But:

> **Lower perplexity does NOT automatically mean a better instruction-following chatbot.**

That's extremely important.

---

# 17. Why perplexity isn't enough for chatbots

Imagine two models.

### Model A

Excellent at predicting ordinary internet text.

### Model B

Excellent at:

```text
following instructions
answering questions
formatting JSON
writing code
refusing unsafe requests
summarizing documents
```

Model B could potentially have worse perplexity while being more useful for your application.

Therefore, for instruction-tuned LLMs, you should evaluate the **actual behavior you care about**.

---

# 18. Exact Match

Suppose the expected answer is:

```text
Paris
```

Model outputs:

```text
Paris
```

Exact match:

```text
1
```

If model outputs:

```text
The answer is Paris.
```

Exact match:

```text
0
```

Even though the answer is semantically correct.

So Exact Match is useful for tasks where the output has a strict expected format.

For example:

```text
Question → answer = 42
```

---

# 19. Token-level F1

Suppose expected answer:

```text
New Delhi
```

Model:

```text
Delhi
```

Exact match says:

```text
wrong
```

But token-level F1 can recognize partial overlap.

This is commonly useful in question-answering evaluation.

---

# 20. ROUGE

ROUGE is commonly associated with summarization.

Suppose reference:

```text
The cat was sleeping on the sofa.
```

Generated:

```text
The cat slept on the sofa.
```

The wording isn't identical, but there is substantial overlap.

ROUGE measures various types of overlap between generated and reference text.

Common variants include:

```text
ROUGE-1
ROUGE-2
ROUGE-L
```

Roughly:

### ROUGE-1

Unigram overlap.

```text
cat
sofa
the
...
```

### ROUGE-2

Two-token sequences.

```text
the cat
on the
the sofa
```

### ROUGE-L

Uses the longest common subsequence.

Again:

> ROUGE is a useful signal, not a perfect measurement of quality.

---

# 21. BLEU

BLEU is widely used for machine translation.

It compares generated translations with reference translations using n-gram precision and includes a brevity penalty.

Example:

```text
Reference:
The cat is sleeping.

Prediction:
The cat sleeps.
```

There is meaningful similarity, but not exact wording.

BLEU attempts to quantify this overlap.

Modern evaluation often uses additional metrics because lexical overlap doesn't fully capture meaning.

---

# 22. Semantic evaluation

Here's a major problem with string-based metrics.

Consider:

```text
Reference:
The car is extremely fast.
```

Model:

```text
The automobile has a very high speed.
```

String overlap might be relatively low.

But semantically:

```text
very similar meaning
```

This is why modern NLP evaluation sometimes uses semantic metrics such as:

- BERTScore
- embedding similarity
- model-based judges
- human evaluation

---

# 23. Human evaluation

For generative LLMs, humans can evaluate things such as:

### Correctness

Is the answer factually correct?

### Relevance

Does it answer the question?

### Helpfulness

Does it actually solve the user's problem?

### Fluency

Is it understandable?

### Instruction following

Did it follow the requested format?

### Safety

Does it behave appropriately for unsafe or sensitive requests?

For example, you could have humans score:

```text
Correctness:       1–5
Relevance:         1–5
Instruction follow: 1–5
```

But human evaluation is:

- expensive
- slower
- potentially subjective
- difficult to reproduce perfectly

So automated + human evaluation is often useful.

---

# 24. LLM-as-a-judge

A powerful modern technique is:

```text
Prompt
  ↓
Model being evaluated
  ↓
Answer
  ↓
Judge LLM
  ↓
score / explanation
```

For example:

```text
Question:
Explain binary search.

Candidate answer:
...

Judge:
Score correctness from 1–5.
Score relevance from 1–5.
Explain the score.
```

You can compare:

```text
Base model
vs
Fine-tuned model
```

using the same evaluation questions.

But LLM judges have limitations:

- judge bias
- preference for certain writing styles
- sensitivity to prompt wording
- possible factual mistakes
- position/order bias
- tendency to favor verbose answers

Therefore, don't blindly trust a judge score.

---

# 25. Pairwise evaluation

Instead of asking:

> "Give this answer a score."

you can ask:

```text
Question

Answer A
Answer B

Which answer better satisfies the criteria?
```

For example:

```text
A = base model
B = LoRA model
```

Then the evaluator compares them.

This is particularly useful for generative models because absolute scoring can be difficult.

But pairwise results should still be interpreted carefully and shouldn't be treated as objective truth.

---

# 26. Task-specific evaluation is the most important thing

Imagine you fine-tune a model specifically for:

> **Generating Dockerfiles.**

Don't primarily evaluate it by asking:

```text
"How fluent is your English?"
```

Instead test:

```text
Input:
Create a Dockerfile for a Python Flask application.

Output:
Dockerfile
```

Then actually test it.

For example:

```text
Generated Dockerfile
       ↓
docker build
       ↓
Did it build?
       ↓
docker run
       ↓
Does application work?
```

That's much more meaningful.

---

# 27. Code generation evaluation

This is especially useful for you because you're interested in coding models.

Suppose the model generates:

```python
def add(a, b):
    return a + b
```

You shouldn't merely compare the text with a reference answer.

Instead:

```text
Generated code
      ↓
Run test cases
      ↓
Pass / fail
```

Example:

```python
assert add(2, 3) == 5
assert add(-1, 1) == 0
assert add(0, 0) == 0
```

If all pass:

```text
Pass
```

This is much more meaningful than exact string matching.

---

# 28. Pass@k

For code generation, you may generate multiple solutions.

Suppose:

```text
k = 10
```

The model generates:

```text
solution 1 → fail
solution 2 → fail
solution 3 → pass
...
```

If at least one of the generated solutions passes, the problem is solved under a pass@k evaluation setup.

So:

```text
Pass@1
```

means roughly:

> probability that one generated solution passes when generating one solution.

```text
Pass@10
```

asks how often at least one of up to ten generated solutions succeeds, under the specific estimator/protocol being used.

---

# 29. Retrieval evaluation

This becomes important when you build RAG systems.

Suppose your knowledge base contains:

```text
Document A
Document B
Document C
...
Document Z
```

User asks:

```text
What is the refund policy?
```

Your retriever returns:

```text
Top 5 documents
```

The question is:

> Did it retrieve the documents containing the relevant information?

Metrics include:

### Recall@k

Did the relevant document appear within the top k?

Example:

```text
Top 5 retrieved:
A
C
F
H
M

Relevant:
F
```

Therefore:

```text
Recall@5 = 1
```

---

### Precision@k

How many retrieved documents are actually relevant?

If:

```text
5 retrieved
2 relevant
```

then:

```text
Precision@5 = 2/5 = 0.4
```

---

# 30. RAG evaluation has two separate parts

This is extremely important.

A RAG system:

```text
Question
   ↓
Retriever
   ↓
Relevant documents
   ↓
LLM
   ↓
Answer
```

You should evaluate:

### Retrieval

Did we retrieve the correct information?

and:

### Generation

Did the LLM correctly use that information?

A system can fail because:

```text
Retriever failed
```

or:

```text
Retriever succeeded
but LLM produced a bad answer
```

These are different problems.

---

# 31. Hallucination evaluation

Suppose you ask:

> "According to this document, what is the refund period?"

Document says:

```text
Refunds are available within 30 days.
```

Model says:

```text
Refunds are available within 60 days.
```

That's a factual error/hallucination relative to the provided source.

For RAG systems, useful evaluation dimensions include:

```text
Faithfulness:
Is the answer supported by retrieved/context documents?

Answer correctness:
Is the answer actually correct?

Context relevance:
Was the retrieved context relevant?

Context recall:
Did retrieval include the necessary information?
```

---

# 32. Evaluation dataset

This is where practical LLM work becomes interesting.

You need an **evaluation dataset**.

Suppose you're building a model that answers questions about your college syllabus.

Create:

```json
[
  {
    "question": "What topics are covered in Unit 3?",
    "answer": "..."
  },
  {
    "question": "What is the prerequisite for this course?",
    "answer": "..."
  }
]
```

Your evaluation dataset should contain examples representative of real usage.

Don't create only easy questions.

Include:

```text
Easy
Medium
Hard
Edge cases
Ambiguous questions
Long inputs
Short inputs
Common mistakes
```

---

# 33. Data leakage

This is one of the biggest evaluation problems.

Suppose:

```text
Training dataset:
Question A → Answer A
```

Then you accidentally put the same question in:

```text
Test dataset
```

Your model gets it right.

You conclude:

> "My model performs extremely well."

But it may have simply memorized it.

That's **data leakage**.

The test set should be independent.

---

# 34. Train/validation/test contamination

Imagine:

```text
Train:
80%

Validation:
10%

Test:
10%
```

You need to ensure related/duplicate examples aren't spread across splits in a way that makes the test artificially easy.

For example:

```text
Train:
"What is recursion?"

Test:
"Explain what recursion means."
```

These may effectively test the same knowledge.

For some datasets, splitting randomly is therefore not sufficient.

---

# 35. Benchmarking

A benchmark is a standardized evaluation setup.

For example, a benchmark may contain thousands of questions covering:

```text
math
reasoning
coding
knowledge
language
...
```

Then you can compare your model's results against other models.

But be careful:

> Benchmark performance doesn't necessarily equal real-world usefulness.

A model can perform well on benchmark questions but perform poorly on your specific application.

---

# 36. Evaluation should happen before and after fine-tuning

This is extremely important for your LoRA work.

Suppose you start with:

```text
Base model
```

Evaluate it:

```text
Base model → 72%
```

Then fine-tune:

```text
LoRA model
```

Evaluate the exact same test set:

```text
LoRA model → 81%
```

Now you have evidence that the fine-tuning changed performance on that evaluation task.

The comparison should be:

```text
              SAME TEST SET

Base model
     ↓
 Evaluation
     ↓
72%

       VS

LoRA model
     ↓
 Evaluation
     ↓
81%
```

Don't change the test set between experiments.

---

# 37. A proper experiment

Suppose you're training a model to generate SQL.

You have:

```text
1,000 examples
```

Split:

```text
800 train
100 validation
100 test
```

Baseline:

```text
Base model
```

Evaluate:

```text
Execution accuracy = 64%
```

Then fine-tune with LoRA.

Evaluate:

```text
LoRA model
Execution accuracy = 78%
```

Then inspect errors.

Maybe:

```text
Simple SQL:        95%
Joins:             80%
Nested queries:    61%
Aggregation:       70%
```

Now you know where the model struggles.

This is much more useful than saying:

> "My model got 78%."

---

# 38. Error analysis

Evaluation doesn't end with a metric.

This is one of the most valuable skills you can develop.

Suppose:

```text
100 test examples

80 correct
20 wrong
```

Don't just write:

```text
Accuracy = 80%
```

Look at the 20 failures.

Categorize them:

```text
5 → hallucination
4 → misunderstood question
3 → formatting problem
3 → missing knowledge
2 → arithmetic mistake
2 → instruction-following failure
1 → context too long
```

Now you know what to improve.

---

# 39. Evaluation is a feedback loop

The real workflow is:

```text
Build model
     ↓
Evaluate
     ↓
Find failures
     ↓
Understand why
     ↓
Improve data/model/training
     ↓
Evaluate again
```

This is much more powerful than:

```text
Train → get score → done
```

---

# 40. Evaluation during SFT

When you're doing supervised fine-tuning:

```text
Base LLM
   ↓
SFT dataset
   ↓
LoRA / QLoRA
   ↓
Fine-tuned model
```

During training, you can monitor:

```text
training loss
validation loss
```

For example:

```text
Epoch    Train Loss    Val Loss

1          2.4          2.5
2          1.9          2.0
3          1.6          1.8
4          1.3          1.9
```

You might select the checkpoint based on validation behavior.

But after training, perform task-specific evaluation.

---

# 41. Evaluation vs validation

These words are often confused.

### Validation

Used **during development**.

```text
"Which configuration should I use?"
```

### Test

Used **after decisions are finalized**.

```text
"How well does my final model perform?"
```

A simple analogy:

```text
Training = studying

Validation = practice exam

Test = final exam
```

---

# 42. Evaluation vs testing

In casual ML discussions, people often use "evaluation" and "testing" interchangeably.

But conceptually:

```text
Evaluation
= broad process of measuring model performance

Testing
= evaluation on a held-out test set
```

Evaluation can include:

```text
validation loss
test metrics
human evaluation
benchmark results
error analysis
robustness tests
safety tests
latency
memory usage
cost
```

---

# 43. Don't evaluate only quality

For a real application, you may care about:

### Quality

Does it produce good answers?

### Speed

How fast?

```text
tokens/sec
latency
```

### Memory

How much VRAM/RAM?

### Cost

How expensive is inference?

### Reliability

Does it consistently work?

### Context handling

Can it handle long documents?

### Safety

Does it behave appropriately?

### Robustness

Does small wording change break it?

A practical evaluation might look like:

```text
Quality       → 84%
Latency       → 120 ms
Memory        → 6.8 GB
Failure rate  → 4%
Context       → 32k tokens
```

---

# 44. Robustness testing

Suppose your model answers:

```text
"What is Docker?"
```

correctly.

Now try:

```text
Can you explain docker?
WHAT IS DOCKER?
Explain Docker in simple words.
What exactly is Docker used for?
```

Does it continue working?

You can deliberately vary:

```text
capitalization
punctuation
wording
question length
irrelevant information
typos
```

This tests robustness.

---

# 45. Edge-case testing

Don't only test normal inputs.

Test:

```text
empty input
very long input
invalid input
ambiguous input
contradictory instructions
unknown questions
malformed data
```

For example:

```text
User:
Explain the policy described in the document.

Document:
[document has no policy]
```

Does your model say:

```text
"I don't have enough information."
```

or invent something?

That is an important evaluation.

---

# 46. Instruction-following evaluation

Suppose the instruction is:

> "Answer in exactly three bullet points."

Model outputs:

```text
paragraph...
paragraph...
paragraph...
```

Even if the content is correct, it failed the instruction.

So you can evaluate:

```text
Content correctness
+
Instruction adherence
```

---

# 47. Structured-output evaluation

Suppose your model must output JSON:

```json
{
  "name": "Aditya",
  "age": 19
}
```

You can automatically test:

```python
import json

try:
    data = json.loads(output)
    valid = True
except:
    valid = False
```

Then evaluate:

```text
JSON validity
Required fields
Correct data types
Correct values
```

This is much better than using a vague "quality score."

---

# 48. A practical evaluation harness

Eventually, you can build something like:

```text
eval.py

Load model
Load test dataset

for each example:

    generate answer

    calculate:
        correctness
        format validity
        latency

    save result

calculate:
    average score
    failure rate

print report
```

Output:

```text
================================
MODEL EVALUATION
================================

Examples:              500

Correctness:            84.2%
Format validity:        97.8%
Average latency:        1.42 sec
Failure rate:            6.1%

================================
ERROR BREAKDOWN
================================

Hallucination:           12
Wrong reasoning:          9
Formatting:               5
Missing knowledge:        4
================================
```

Now you have an actual evaluation system.

---

# 49. Hugging Face evaluation

Since you're learning Hugging Face, you'll encounter libraries/tools around:

```text
Transformers
Datasets
Evaluate
TRL
PEFT
```

The general workflow is:

```text
datasets
   ↓
preprocess
   ↓
train
   ↓
evaluate
   ↓
metrics
```

A common conceptual pattern looks like:

```python
from evaluate import load

metric = load("accuracy")

predictions = [...]
references = [...]

results = metric.compute(
    predictions=predictions,
    references=references
)

print(results)
```

The exact metric depends on your task.

For generation, you often need custom evaluation logic rather than blindly applying classification metrics.

---

# 50. Example: evaluating a classifier

Suppose:

```python
predictions = [1, 0, 1, 1, 0]
references  = [1, 0, 0, 1, 0]
```

Compare them:

```text
Prediction   Actual

1            1       ✓
0            0       ✓
1            0       ✗
1            1       ✓
0            0       ✓
```

Therefore:

```text
4 / 5 correct

Accuracy = 0.8
```

---

# 51. Example: evaluating an LLM

Suppose your test dataset:

```json
[
  {
    "question": "What is a pointer?",
    "reference": "A pointer stores the memory address..."
  },
  {
    "question": "What is recursion?",
    "reference": "Recursion is when..."
  }
]
```

Your model generates:

```text
Answer 1
Answer 2
```

You could evaluate using several approaches:

```text
Reference-based:
    ROUGE
    BERTScore

Semantic:
    embedding similarity

Model-based:
    LLM judge

Human:
    correctness/relevance

Task-specific:
    custom tests
```

For a teaching chatbot, for example, I would care much more about:

```text
correctness
relevance
instruction following
hallucination
```

than simple word overlap.

---

# 52. One important trap: metric gaming

Suppose you're optimizing for ROUGE.

You might accidentally create a model that produces answers that resemble reference answers but aren't actually better.

This is called, broadly, **Goodhart's law**:

> When a measure becomes a target, it can stop being a good measure.

Therefore:

```text
Metric ≠ reality
```

A metric is a proxy for the property you actually care about.

---

# 53. Evaluation should match your objective

Think of it like this:

```text
REAL GOAL
   ↓
"What does good performance mean?"
   ↓
Define measurable criteria
   ↓
Build evaluation dataset
   ↓
Choose metrics
   ↓
Run evaluation
   ↓
Inspect errors
```

For example:

### Goal: coding model

Use:

```text
Unit-test pass rate
Compilation rate
Pass@k
```

### Goal: RAG assistant

Use:

```text
Retrieval recall
Context relevance
Answer correctness
Faithfulness
```

### Goal: summarization

Use:

```text
ROUGE/BERTScore
factuality
human evaluation
```

### Goal: classification

Use:

```text
Precision
Recall
F1
Confusion matrix
```

---

# 54. Confusion matrix

For classification, a confusion matrix is extremely useful.

Suppose you classify:

```text
cat
dog
horse
```

You might get:

| Actual ↓ / Predicted → | Cat | Dog | Horse |
|---|---:|---:|---:|
| Cat | 90 | 5 | 5 |
| Dog | 10 | 80 | 10 |
| Horse | 2 | 8 | 90 |

You immediately see:

```text
Dog ↔ Horse
```

is more problematic than:

```text
Cat ↔ Horse
```

So aggregate accuracy alone hides useful information.

---

# 55. Distribution matters

Suppose your test set contains:

```text
90% easy questions
10% hard questions
```

Your model gets:

```text
95% overall
```

That might sound excellent.

But perhaps:

```text
Easy → 99%
Hard → 59%
```

If your real users mostly ask hard questions, the 95% number isn't representative.

Therefore evaluate by categories:

```text
Easy
Medium
Hard

Short
Long

Known
Unknown

Domain A
Domain B
```

---

# 56. Confidence and calibration

Some models can produce confidence estimates.

For classification:

```text
Prediction:
Cat

Confidence:
99%
```

But suppose the model is correct only 70% of the time when it says 99%.

That's poor calibration.

A well-calibrated model should roughly satisfy:

```text
80% confidence
→ correct about 80% of the time
```

Calibration matters in applications where confidence influences decisions.

---

# 57. Evaluation for your LoRA/QLoRA learning path

Since you're learning:

```text
Transformer
 ↓
Tokenization
 ↓
Attention
 ↓
Causal LM
 ↓
Embeddings
 ↓
SFT
 ↓
LoRA / PEFT
 ↓
QLoRA
 ↓
Datasets + cleaning
 ↓
Evaluation
```

you should understand evaluation as the step that tells you:

> **"Did all the work above actually improve the model for my target task?"**

Your practical workflow should eventually look like:

```text
1. Choose base model

2. Create dataset

3. Clean dataset

4. Split dataset

       ┌──────────────┐
       │              │
       ↓              ↓
    Train         Validation
       ↓              ↓
       └──────┬───────┘
              ↓
        LoRA / QLoRA
              ↓
       Fine-tuned model
              ↓
       Evaluate on TEST
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
 Metrics   Error      Human/
           analysis   LLM judge
    ↓         ↓         ↓
    └─────────┼─────────┘
              ↓
       Improve experiment
```

---

# 58. The experiment you should actually do

You can learn evaluation extremely well with one small project.

### Project

Fine-tune a small model to answer questions about a specific technical subject.

Create:

```text
1000 examples
```

Split:

```text
800 train
100 validation
100 test
```

Then:

### Step 1 — Evaluate base model

Run the **base model** on the 100 test questions.

Record:

```text
correctness
format
hallucinations
```

---

### Step 2 — Fine-tune

Use:

```text
LoRA/QLoRA
```

---

### Step 3 — Evaluate again

Use **the exact same test set**.

Compare:

```text
Base model
vs
Fine-tuned model
```

---

### Step 4 — Categorize errors

For every failed example:

```text
What went wrong?
```

Record:

```text
knowledge error
reasoning error
hallucination
instruction failure
format failure
```

---

### Step 5 — Improve

Maybe you discover:

```text
Model performs poorly on long questions.
```

Then investigate.

Maybe:

```text
dataset doesn't contain enough long examples
```

So improve the dataset.

Run the experiment again.

This teaches you much more than simply calculating an F1 score.

---

# 59. Your evaluation checklist

Whenever you train a model, ask:

### Dataset

- Is my test set separate?
- Is there leakage?
- Does it represent real usage?
- Are there enough difficult examples?

### Metrics

- What does "good" actually mean?
- Am I using the right metric?
- Is one metric sufficient?

### Baseline

- How does the original model perform?
- Did fine-tuning actually improve anything?

### Validation

- Is validation performance improving?
- Is the model overfitting?

### Test

- What is the final performance?
- Did I touch the test set during development?

### Error analysis

- What kinds of mistakes occur?
- Are errors concentrated in particular categories?

### Real-world performance

- Is it useful?
- Is it fast enough?
- Is it reliable?
- Does it hallucinate?
- Does it follow instructions?

---

# 60. The most important concepts to remember

If you remember only these, you understand the core of evaluation:

```text
1. Training ≠ evaluation

2. Training data ≠ test data

3. Validation helps you develop the model.

4. Test data gives the final held-out measurement.

5. Loss tells you about the training objective;
   it doesn't completely describe usefulness.

6. Choose metrics according to the task.

7. Accuracy isn't always enough.

8. For LLMs, task-specific evaluation is extremely important.

9. Automated metrics are imperfect proxies.

10. Human/LLM evaluation can complement automated metrics.

11. Always inspect individual failures.

12. Compare against a baseline.

13. Keep the test set fixed while comparing models.

14. Watch for data leakage.

15. Evaluation is a loop:
   measure → analyze → improve → measure again.
```

## The mental model I want you to have

Think of evaluation as a **scientific experiment**:

```text
                HYPOTHESIS
                    │
                    ▼
       "Fine-tuning will improve X"
                    │
                    ▼
             SAME TEST SET
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      BASE MODEL          FINE-TUNED
          │                   │
          ▼                   ▼
      EVALUATE             EVALUATE
          │                   │
          └─────────┬─────────┘
                    ▼
              COMPARE RESULTS
                    │
                    ▼
               ERROR ANALYSIS
                    │
                    ▼
             IMPROVE SYSTEM
```

That is the core of **evaluation in practical LLM engineering**.

And when you eventually build your own Hugging Face projects, don't think of evaluation as **"calculate a score at the end."** Think of it as the mechanism that tells you **whether your model actually works, where it fails, and what you should change next.**

