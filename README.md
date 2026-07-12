# ZeroCredits

ZeroCredits is an adaptive multi-model inference layer for mixed AI workloads. It classifies each request, ranks every model available at runtime, selects the most suitable execution path, validates the response, and preserves the required input/output contract.

The system combines:

- local CPU inference for short, narrow tasks;
- hosted models for reasoning, extraction, summarization, and code;
- category-specific model ranking;
- bounded remote concurrency;
- task-specific prompts and output budgets;
- ranked fallbacks;
- exact-prompt runtime caching;
- ordered JSON output.

ZeroCredits is not a single-model wrapper. It is a control layer that matches each workload to the model family most likely to produce a correct, complete, and well-formatted answer.

---

## Architecture

```text
tasks.json
    |
    v
Input validation
    |
    v
Task classification
    |
    v
Model ranking
    |
    +-----------------------------+
    |                             |
    v                             v
Serial local lane          Parallel remote lane
Qwen2.5 1.5B               All allowed hosted models
sentiment and tone         ranked by task category
    |                             |
    +-------------+---------------+
                  |
                  v
        Validation and cleanup
                  |
                  v
       Ordered results.json
```

The pipeline separates planning from execution.

1. Tasks are loaded and validated.
2. Each prompt is classified into a supported category.
3. Every allowed hosted model is ranked for that category.
4. Sentiment tasks are assigned to the local lane when eligible.
5. Other tasks are sent to the hosted lane.
6. Failed local tasks fall back to the ranked remote path.
7. Responses are validated, cached, and restored to input order.
8. The final result file contains one answer for every task.

---

## Supported task categories

ZeroCredits currently supports eight task families:

| Category | Typical workload |
|---|---|
| Factual | Definitions, explanations, comparisons |
| Math | Arithmetic, algebra, probability, word problems |
| Logic | Ordering, deduction, constraint reasoning |
| Summarization | Sentence, bullet, title, and length-constrained summaries |
| NER | Named-entity extraction and structured output |
| Sentiment | Sentiment, tone, and opinion classification |
| Code debugging | Repairing incorrect Python, JavaScript, or other code |
| Code generation | Writing functions, queries, classes, and algorithms |

---

## Model routing

All models provided through `ALLOWED_MODELS` remain eligible. The router changes their priority by task category; it does not remove them from the candidate set.

### Gemma

Gemma is the quality-first family for non-code workloads in this project.

The routing policy ranks Gemma first for six of the eight supported categories:

- factual answering;
- summarization;
- named-entity extraction;
- sentiment;
- mathematics;
- logic.

This means Gemma is the preferred family for 75% of the task taxonomy.

The project recognizes multiple Gemma variants:

- Gemma 4 26B A4B IT;
- Gemma 4 31B IT;
- Gemma 4 31B IT NVFP4.

The 26B variant is suited to concise general-language and structured-output tasks. The 31B variants are preferred for stronger reasoning and instruction adherence.

### Kimi

Kimi is ranked first for:

- code generation;
- code debugging.

The routing policy treats code as a specialized workload rather than forcing a general-purpose model to handle it.

### MiniMax

MiniMax acts as a broad general-purpose fallback. It is useful across several categories, but local runs showed that hidden reasoning can consume small completion budgets before a visible answer is produced. For that reason, MiniMax receives additional provider-side headroom.

### Local Qwen

Qwen2.5 1.5B Instruct runs locally for sentiment and tone classification.

The local lane is deliberately narrow:

- one task at a time;
- no local batching;
- no local task parallelism;
- two llama.cpp inference threads;
- remote work continues separately.

This reduces remote usage without placing complex reasoning tasks on a small CPU model.

---

## Internal calibration findings

The routing strategy was calibrated using a 40-task mixed workload.

Verified findings from local runs include:

| Metric | Result |
|---|---:|
| Internal calibration tasks | 40 |
| Supported task categories | 8 |
| Categories ranking Gemma first | 6 of 8 |
| Gemma quality-first coverage | 75% |
| Stable validation accuracy | 94.7% |
| Stable token usage | 7,909 |
| Aggressively trimmed accuracy | 84.2% |
| Aggressively trimmed token usage | 7,004 |
| Token reduction | 11.4% |
| Accuracy reduction | 10.5 percentage points |

The main conclusion was that small token savings can create a disproportionate quality loss when budgets are reduced too aggressively. ZeroCredits therefore protects answer quality first and optimizes cost through routing, caching, specialization, and category-aware limits.

The available artifacts support category-level Gemma preference and run-level statistics. They do not contain a complete model-by-model exact-score table for all 40 tasks, so no unsupported per-model accuracy percentage is claimed.

---

## Execution model

### Local lane

Local tasks execute serially in the main process.

```text
sentiment_01
sentiment_02
sentiment_03
...
```

`LOCAL_THREADS=2` means that one local inference uses two CPU threads. It does not mean that two local tasks execute simultaneously.

### Remote lane

Hosted requests use a bounded `ThreadPoolExecutor`.

This is appropriate because hosted inference is largely network-bound. Bounded concurrency reduces wall-clock time without creating unbounded API pressure.

Local inference and hosted requests may overlap, but only remote work is submitted to the executor.

---

## Research-derived concepts

### Model routing

Selecting a model separately for each request instead of using one fixed model for every workload.

### Model cascade

Trying a preferred model first and moving to another ranked model when the first attempt fails or produces an invalid response.

### Adaptive inference

Changing the execution path according to task category, expected complexity, latency, and output format.

### Cost-quality optimization

Selecting the least expensive execution path likely to meet the required answer quality.

### System-level mixture of experts

Using several complete models as specialists, with a router deciding which specialist should handle each request.

### Bounded concurrency

Allowing several remote requests to run simultaneously while enforcing a fixed maximum number of workers.

### Backpressure

Preventing the application from creating more simultaneous work than the runtime or provider can safely support.

### Token budget

The maximum number of output tokens available to a model call.

### Visible-answer budget

The expected size of the final user-facing response. Some providers need extra generation headroom because reasoning may consume tokens before visible content appears.

### Quantization

Reducing model-weight precision to lower memory use and improve local inference speed.

### GGUF

A model packaging format commonly used by llama.cpp-compatible runtimes.

### Context window

The maximum combined prompt and generated text that a model can process in one request.

### Prompt specialization

Using category-specific system prompts rather than one generic instruction for every workload.

### Output validation

Checking whether a response is present, complete, correctly formatted, and safe to accept.

### Runtime cache

Reusing an answer for an exact duplicate prompt during the same execution.

---

## Tradeoffs

### Qwen2.5 1.5B instead of 0.5B

The 0.5B model used less memory and completed faster, but it was less reliable at instruction following and structured sentiment output. The 1.5B model provided a better quality-to-resource balance.

### No local batching

Each sentiment task is sent to Qwen independently.

Benefits:

- simpler parsing;
- fewer cross-task formatting failures;
- clearer logs;
- easier fallback.

Cost:

- repeated prompt-processing overhead.

### Serial local execution

Only one local generation runs at a time.

Benefits:

- both CPUs are available to one inference;
- avoids contention;
- predictable memory usage.

Cost:

- local tasks do not overlap with one another.

### Parallel remote execution

Hosted requests are processed concurrently.

Benefits:

- reduced total runtime;
- network wait is overlapped;
- limited CPU impact.

Cost:

- additional API pressure;
- more concurrent memory and connection usage.

### All allowed models remain eligible

The router ranks the complete runtime model set.

Benefits:

- resilient to runtime availability;
- no hard dependency on one hosted model;
- easy to add or remove models through configuration.

Cost:

- model aliases and ranking rules must remain maintained.

### Bounded fallbacks

A failed model call may move to the next ranked model.

Benefits:

- higher completion reliability;
- fewer missing answers.

Cost:

- extra latency and tokens when fallbacks occur.

### Category-specific output budgets

Different tasks receive different output limits.

Benefits:

- short tasks avoid wasted tokens;
- code and reasoning tasks receive enough space.

Cost:

- budgets require calibration;
- overly aggressive limits can truncate valid answers.

---

## Input format

The default input path is:

```text
/input/tasks.json
```

Example:

```json
[
  {
    "task_id": "task_001",
    "prompt": "Calculate 15% of 200."
  },
  {
    "task_id": "task_002",
    "prompt": "Fix this Python function: def add(a, b): return a - b"
  }
]
```

Every task must contain:

- a non-empty string `task_id`;
- a non-empty string `prompt`.

---

## Output format

The default output path is:

```text
/output/results.json
```

Example:

```json
[
  {
    "task_id": "task_001",
    "answer": "30"
  },
  {
    "task_id": "task_002",
    "answer": "def add(a, b):\n    return a + b"
  }
]
```

Output order matches input order, even though hosted requests may finish in a different order.

---

## Build

Build the Linux AMD64 image:

```bash
docker build \
  --platform linux/amd64 \
  -t docker.io/1bodhi/zerocredits:0.19 .
```

The image includes the local Qwen2.5 1.5B GGUF model and the Python runtime.

---

## Run

```bash
docker run --rm \
  --platform linux/amd64 \
  --cpus="2" \
  --memory="4g" \
  --memory-swap="4g" \
  -e LOCAL_THREADS=2 \
  -e REMOTE_WORKERS=6 \
  -e REMOTE_MODEL_ATTEMPTS=2 \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  docker.io/1bodhi/zerocredits:0.19
```

---

## Environment variables

| Variable | Typical value | Purpose |
|---|---:|---|
| `FIREWORKS_API_KEY` | required | Authentication for hosted inference |
| `FIREWORKS_BASE_URL` | provider URL | Hosted inference endpoint |
| `ALLOWED_MODELS` | runtime supplied | Complete model candidate set |
| `LOCAL_MODEL_PATH` | image defined | Path to the local GGUF file |
| `LOCAL_THREADS` | `2` | Threads used by one local inference |
| `REMOTE_WORKERS` | `4` to `6` | Maximum simultaneous hosted requests |
| `REMOTE_MODEL_ATTEMPTS` | `1` to `2` | Ranked model attempts per task |
| `FIREWORKS_TIMEOUT_SECONDS` | `10` to `15` | Per-request timeout |
| `OMP_NUM_THREADS` | `2` | OpenMP thread limit |
| `OPENBLAS_NUM_THREADS` | `1` | Prevents BLAS oversubscription |
| `MKL_NUM_THREADS` | `1` | Prevents MKL oversubscription |
| `NUMEXPR_NUM_THREADS` | `1` | Prevents NumExpr oversubscription |

`LOCAL_BATCH_SIZE` is not used because local tasks are not batched.

---

## Recommended operating profiles

### Balanced

```bash
LOCAL_THREADS=2
REMOTE_WORKERS=6
REMOTE_MODEL_ATTEMPTS=2
```

Use when wall-clock time matters and one fallback is acceptable.

### Stable

```bash
LOCAL_THREADS=2
REMOTE_WORKERS=4
REMOTE_MODEL_ATTEMPTS=1
```

Use when lower request pressure and predictable behavior matter more than maximum throughput.

### Conservative API usage

```bash
LOCAL_THREADS=2
REMOTE_WORKERS=3
REMOTE_MODEL_ATTEMPTS=1
```

Use when minimizing simultaneous hosted requests is the priority.

---

## Local model behavior

The local model is loaded once and cached for the duration of the process.

Recommended llama.cpp configuration:

```python
Llama(
    model_path=model_path,
    n_ctx=1024,
    n_threads=2,
    n_threads_batch=2,
    n_batch=64,
    use_mmap=True,
    use_mlock=False,
    verbose=False,
)
```

The message:

```text
n_ctx_per_seq (1024) < n_ctx_train (32768)
```

is informational. It means the runtime is using a smaller context window than the model supports.

---

## Failure handling

A task may move to another model when:

- the local model is not eligible;
- local model loading fails;
- local generation raises an exception;
- a hosted request times out;
- a provider returns no visible answer;
- the response is empty or invalid;
- a model reaches its output limit without producing an acceptable completion.

Failures are isolated per task so the application can still write a complete output file.

---

## Caching

The runtime cache uses the exact prompt as its key.

A repeated prompt in the same batch reuses the earlier answer and avoids another local or hosted call.

The cache is intentionally process-local:

- no persistent storage;
- no cross-run stale data;
- no semantic similarity matching;
- no accidental reuse across different prompts.

---

## Logging

Typical logs include:

```text
INFO tasks=64 local_sentiment=8 remote=56 remote_workers=6
INFO task=sentiment_01 category=sentiment route=local max_tokens=32
INFO task=codegen_01 category=code_generation route=remote model=... max_tokens=640
INFO task=task_004 route=cache
WARNING task=logic_06 category=logic route=remote model=... failed: ...
```

These fields make routing and fallback behavior inspectable without exposing API credentials.

---

## Known limitations

- Classification is heuristic rather than learned.
- Exact-prompt caching does not detect semantically equivalent requests.
- Small local models may still violate strict JSON or mixed-sentiment instructions.
- Hosted reasoning models may consume output budget before visible content.
- A response that reaches `finish_reason="length"` may be incomplete even when text is present.
- The 40-task calibration set is useful for routing decisions but is not a comprehensive benchmark.
- Model names and aliases may need updates when the runtime model portfolio changes.

---

## Recommended next improvements

1. Reject all length-truncated non-code responses.
2. Strip Markdown fences from JSON-only outputs.
3. Validate requested word, sentence, and bullet counts.
4. Add category-level answer validators.
5. Record latency and token usage per model family.
6. Build a larger held-out calibration set.
7. Learn routing preferences from observed outcomes rather than maintaining only static rankings.
8. Add structured health metrics for local and hosted lanes.

---

## Method summary

ZeroCredits follows a simple operating sequence:

```text
classify
  -> rank all allowed models
  -> choose local or hosted execution
  -> apply category-specific instructions
  -> execute with bounded resources
  -> validate the result
  -> fall back when necessary
  -> preserve input order
```

The resulting system balances answer quality, throughput, token usage, and resource constraints without forcing every task through the same model.
