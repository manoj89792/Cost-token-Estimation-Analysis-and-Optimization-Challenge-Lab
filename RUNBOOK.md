# Detailed Lab Runbook: Cost/Token Estimation, Analysis, and Optimization Challenge Lab

This runbook documents every step, code cell, cost-model calculation, ledger breakdown, and optimization decision used to complete the **Cost/Token Estimation, Analysis, and Optimization: Challenge Lab** on Google Cloud Agent Platform Workbench.

---

## Lab Constraints & Final Challenge Target

In the final challenge (Part 6), the **Code Assistant Agent** is evaluated on **16 held-out developer queries** (`assets/eval_holdout.jsonl`) under the following production requirements:

| Constraint | Requirement |
| :--- | :--- |
| **Cost Budget** | Average cost per query **$\le \$0.0135\text{ USD}$** |
| **Quality Floor** | Average reference-anchored correctness score **$\ge 0.90$** (normalized from a 1–5 LLM judge rubric) |
| **Coordinator Model Tier** | Must be **`gemini-3.1-pro-preview`** or **`gemini-3.5-flash`** (`gemini-3.1-flash-lite` is prohibited on the Coordinator seat) |
| **Sub-Agent Model Tiers** | Unrestricted (`gemini-3.1-pro-preview`, `gemini-3.5-flash`, or `gemini-3.1-flash-lite`) |
| **Embedding Model** | Must remain pinned to **`gemini-embedding-001`** |
| **Deliverable** | Upload `/home/jupyter/final_metrics.json` with `"passed": true` to `gs://<PROJECT_ID>-metrics-result/results/final_metrics.json` |

---

## Step 1: Access JupyterLab & Verify Lab Files (Tasks 1 & 2)

### 1.1 Open JupyterLab on Agent Platform Workbench
1. Sign in to the Google Cloud Console using your temporary lab credentials.
2. Navigate to **Agent Platform > Notebooks > Workbench**.
3. Click **Open JupyterLab** next to the **`generative-ai-jupyterlab`** instance.

### 1.2 Verify (or Stage) Lab Files in `/home/jupyter`
Confirm the following files exist in the JupyterLab file browser:
- `code_assistant_cost_optimization.ipynb`
- `code_assistant/` (`__init__.py`, `agents.py`, `data.py`, `display.py`, `ledger.py`, `quality.py`, `retrieval.py`, `vais.py`)
- `assets/` (`docs.jsonl`, `stackoverflow.jsonl`, `eval_dataset.jsonl`, `eval_holdout.jsonl`)

If any files are missing, open a Terminal in JupyterLab and run:
```bash
export PROJECT_ID=$(gcloud config get-value project)
gcloud storage cp -r "gs://${PROJECT_ID}-bucket/*" ~/
```

---

## Step 2: Part 0 — Setup, Cost Model, Quality Rubric & Agent Search Provisioning (Tasks 3 & 4)

### 2.1 Install Dependencies & Restart Kernel
Run Cell 1 in `code_assistant_cost_optimization.ipynb`:
```python
%pip install -q google-adk==2.3.0 "google-cloud-aiplatform[evaluation]>=1.158.0" "google-genai>=2.10.0" "chromadb>=1.5.9" "google-cloud-discoveryengine>=0.20.0" pandas matplotlib
```
Then run Cell 2 to restart the kernel so the newly installed packages take effect:
```python
import os
if not os.getenv("IS_TESTING"):
    import IPython
    IPython.Application.instance().kernel.do_shutdown(True)
```

### 2.2 Configure Environment & Baseline Model Map
Run Cells 3 and 4:
```python
PROJECT_ID = !(gcloud config get-value project)
PROJECT_ID = PROJECT_ID[0]
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
LOCATION = "global"           # Gemini 3 family serves from the global endpoint
EVAL_LOCATION = "us-central1" # Gen AI evaluation service is regional

MODELS = {
    "coordinator": "gemini-3.1-pro-preview",
    "docs": "gemini-3.5-flash",
    "community": "gemini-3.5-flash",
    "embedding": "gemini-embedding-001",
}
THINKING = "HIGH"
```

### 2.3 Define the Cost Model (`PRICING` & `cost_of`)
Run Cell 5. Key pricing rules in this lab:
- **Thought (reasoning) tokens bill at the model's output token rate**.
- **`prompt_token_count` from the Gemini API includes `cached_content_token_count`**, so `code_assistant/agents.py` records `input_tokens = max(0, prompt - cached)` and `cached_tokens = cached` disjointly.
- **Agent Search (`vais_query`)** bills a flat **$1.50 per 1,000 queries** (`$0.0015` per lookup), regardless of `top_k`.
- **Embeddings (`gemini-embedding-001`)** bill **$0.15 per 1M input tokens** (one-time index build queries with `query_id="__index__"` are excluded from `cost_per_query()`).

| Resource / Model | Input ($ / 1M) | Cached ($ / 1M) | Output & Thought ($ / 1M) | Flat / Per-Unit Rate |
| :--- | :---: | :---: | :---: | :---: |
| `gemini-3.1-pro-preview` | `$2.00` | `$0.20` | `$12.00` | — |
| `gemini-3.5-flash` | `$1.50` | `$0.15` | `$9.00` | — |
| `gemini-3.1-flash-lite` | `$0.25` | `$0.025` | `$1.50` | — |
| `gemini-embedding-001` | `$0.15` | — | — | `$0.00015` / 1k tokens |
| `vais_query` (Agent Search) | — | — | — | `$1.50` / 1k queries (`$0.0015`/query) |

### 2.4 Provision Managed Agent Search & Local ChromaDB Index (Task 4 — AT ID: 1)
Run Cell 8 to provision the Discovery Engine data store and search engine:
```python
VAIS_LOCATION, DS_ID, ENGINE_ID = "global", "code_assistant-docs", "code_assistant-docs-engine"
DOCS_SC = vais.ensure(PROJECT, VAIS_LOCATION, DS_ID, ENGINE_ID, load_jsonl("assets/docs.jsonl"))
print("DocsAgent backend: Agent Search engine", ENGINE_ID)

so        = load_jsonl("assets/stackoverflow.jsonl")
dataset   = load_jsonl("assets/eval_dataset.jsonl")    # 15 queries
holdout   = load_jsonl("assets/eval_holdout.jsonl")    # 16 held-out queries
community = build_index(so, MODELS["embedding"], PROJECT, LOCATION, name="community")
print("indexed %d Stack Overflow posts into ChromaDB" % len(so))
```

**Output:**
```text
  created data store: code_assistant-docs
  imported 20 documents
  created engine: code_assistant-docs-engine
DocsAgent backend: Agent Search engine code_assistant-docs-engine
indexed 20 Stack Overflow posts into ChromaDB
```

> **Checkpoint 1 (Task 4)**: Click **Check my progress** in the lab instructions to verify **Provision the Agent Search resources**.

### 2.5 Inspect Single-Query Cost Breakdown
Running the single-query demo (`"I get a KeyError reading a dict in Python. How do I fix it?"`) with the baseline configuration produces 8 billable events totaling **`$0.0236`**:

| Event # | Agent | Call Type | Input Tokens | Output Tokens | Thought Tokens | Retrieval Queries | Embed Tokens | Cost (USD) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `Coordinator` | `model` | 157 | 38 | 194 | 0 | 0 | `$0.00309800` |
| 1 | `DocsAgent` | `model` | 99 | 19 | 37 | 0 | 0 | `$0.00065250` |
| 2 | `DocsAgent` | `retrieval` | 0 | 0 | 0 | 1 | 0 | `$0.00150000` |
| 3 | `CommunityAgent` | `model` | 101 | 23 | 65 | 0 | 0 | `$0.00094350` |
| 4 | `CommunityAgent` | `embedding` | 0 | 0 | 0 | 0 | 7 | `$0.00000105` |
| 5 | `DocsAgent` | `model` | 349 | 318 | 61 | 0 | 0 | `$0.00393450` |
| 6 | `CommunityAgent` | `model` | 378 | 395 | 143 | 0 | 0 | `$0.00540900` |
| 7 | `Coordinator` | `model` | 1103 | 346 | 142 | 0 | 0 | `$0.00806200` |

---

## Step 3: Part 1 — Measure the Baseline

Run the Part 1 cell across all 15 queries in `assets/eval_dataset.jsonl`:
- **Configuration**: `Coordinator = gemini-3.1-pro-preview` (`thinking = HIGH`), `DocsAgent = gemini-3.5-flash`, `CommunityAgent = gemini-3.5-flash`, `top_k = 3`.
- **Measured Result**:
  - **Cost per query**: **`$0.0290`** (`$0.02900412`)
  - **Quality score**: **`0.93`** (`0.9333`)
- **Key Takeaway**: The `Coordinator` (`gemini-3.1-pro-preview` at `$2.00/1M` input and `$12.00/1M` output/thought tokens with `HIGH` thinking) dominates the total spend, followed by the two `gemini-3.5-flash` sub-agents.

---

## Step 4: Part 2 — Lever 1: Model Selection & Tiering

### 4.1 Lever 1a: Tier Coordinator from Pro to Flash (`thinking = HIGH`)
```python
TIERED_MODELS = {**MODELS, "coordinator": "gemini-3.5-flash"}   # was: gemini-3.1-pro-preview
```
- **Measured Result**:
  - **Cost per query**: **`$0.0185`** (`$0.01854996`, **`-36%`** vs. baseline)
  - **Quality score**: **`0.95`** (`0.9500`, **`+0.02`** vs. baseline)
- **Per-Agent Spend (15 queries)**:
  - `Coordinator`: `17,320` input, `4,952` thought, `7,233` output -> `$0.135645`
  - `CommunityAgent`: `6,848` input, `2,891` thought, `4,468` output -> `$0.076523`
  - `DocsAgent`: `5,430` input, `1,994` thought, `2,610` output -> `$0.066081`

### 4.2 Lever 1b: Tier Coordinator to Flash-Lite (`thinking = HIGH`)
```python
LITE_MODELS = {**MODELS, "coordinator": "gemini-3.1-flash-lite"}
```
- **Measured Result**:
  - **Cost per query**: **`$0.0127`** (`$0.01266497`, **`-56%`** vs. baseline)
  - **Quality score**: **`0.88`** (`0.8750`, **`-0.06`** vs. baseline — **falls below the `0.90` quality floor!**)
- **Why Flash-Lite Fails on the Coordinator**:
  - Under `HIGH` thinking, `gemini-3.1-flash-lite` generated **`15,132` thought tokens** (more than 3x Flash's `4,952`), occasionally over-calling sub-agents or failing to synthesize multi-part answers cleanly.

---

## Step 5: Part 3 — Lever 2: Context Caching

Run the standalone Context Caching benchmark on `gemini-3.5-flash` with an 8,409-token system prompt prefix:
```python
cache = client.caches.create(
    model="gemini-3.5-flash",
    config=types.CreateCachedContentConfig(system_instruction=big_prefix, ttl="120s")
)
r = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Name your role in 5 words.",
    config=types.GenerateContentConfig(cached_content=cache.name)
)
```
**Output:**
```text
prompt tokens: 8409, served from cache: 8402
input cost: without cache $0.012613  ->  with cache $0.001271  (90% cheaper)
```
- **Lesson**: Context caching reduces cached input token rates by **90%** (`$1.50/1M -> $0.15/1M` on Flash), but requires a large repeated prefix above the context-cache minimum token threshold. Because Code Assistant Agent's static system prompt is only ~150 tokens, context caching is not applicable to its short instruction prefix.

---

## Step 6: Part 4 — Lever 3: Reasoning & Retrieval Token Reduction

### 6.1 Lever 3a: Lower Coordinator Thinking Level to `LOW`
```python
THINKING_LOW = "LOW"   # was: "HIGH"
coord, ledger = build(TIERED_MODELS, thinking_level=THINKING_LOW)
```
- **Measured Result**:
  - **Cost per query**: **`$0.0124`** (`$0.01238952`, **`-57%`** vs. baseline)
  - **Quality score**: **`0.93`** (`0.9333`, **`+0.00`** vs. baseline)
- **Why it works**: Coordinator `thought_tokens` dropped from `4,952` (`HIGH`) down to `1,898` (`LOW`), cutting Coordinator spend from `$0.1356` to `$0.0924` without hurting synthesis quality.

### 6.2 Lever 3b: Stack Retrieval Breadth Trim (`top_k = 2`)
```python
TOP_K = 2   # was: 3 snippets per lookup
coord, ledger = build(TIERED_MODELS, thinking_level=THINKING_LOW, top_k=TOP_K)
```
- **Measured Result**:
  - **Cost per query**: **`$0.0140`** (`$0.01402861`, **`-52%`** vs. baseline)
  - **Quality score**: **`0.90`** (`0.9000`, **`-0.03`** vs. baseline)

---

## Step 7: Part 5 — Compare Configurations & Semantic Response Cache

### 7.1 Scorecard Summary (Development Set)
```text
                           configuration  cost_per_query    quality  cost_vs_baseline quality_vs_baseline
0             baseline (Pro coordinator)      0.02900412 0.93330000               +0%               +0.00
1       lever 1a: tiering (Flash @ HIGH)      0.01854996 0.95000000              -36%               +0.02
2  lever 1b: tiering (Flash-Lite @ HIGH)      0.01266497 0.87500000              -56%               -0.06
3         lever 3a: thinking LOW (Flash)      0.01238952 0.93330000              -57%               +0.00
4       lever 3b: thinking LOW + top_k 2      0.01402861 0.90000000              -52%               -0.03
```

### 7.2 Bonus: Semantic Response Cache
Running the `SemanticCache` cells seeds the cache with `"How do I fix a KeyError in a Python dict?"` and tests a paraphrase (`"fixing a KeyError when reading a dict"`) and an unrelated query (`"how do I center a div in CSS"`):
```text
CACHE HIT : 'fixing a KeyError when reading a dict'
            served the stored answer 'Use dict.get(key, default).' for the cost of one embedding call
CACHE MISS: 'how do I center a div in CSS'
            no similar entry; this query would run the full multi-agent pipeline
```

---

## Step 8: Part 6 — Final Challenge Solution & Verification (Task 5 — AT ID: 2)

### 8.1 Why Standard Levers Alone Are Not Enough on the Holdout Set
In Lever 3a (`Coordinator = gemini-3.5-flash`, `Sub-Agents = gemini-3.5-flash`, `thinking = LOW`), the two sub-agents (`DocsAgent` and `CommunityAgent`) still run on `gemini-3.5-flash` (`$1.50/1M` input, `$9.00/1M` output/thought) and account for **over 50% of the total bill** (`$0.0935` out of `$0.1858`), putting the configuration right on the edge of the `$0.0135/query` budget on the more complex 16-query holdout set.

### 8.2 The Winning Optimization Strategy
1. **Keep `Coordinator` on `gemini-3.5-flash`**: Complies with the architectural rule forbidding `gemini-3.1-flash-lite` on the Coordinator and preserves high-quality multi-part synthesis.
2. **Tier `DocsAgent` and `CommunityAgent` to `gemini-3.1-flash-lite`**:
   - Sub-agents perform a narrow, single-tool lookup and summarize retrieved text.
   - `gemini-3.1-flash-lite` is **6x cheaper** on both input (`$0.25/1M` vs. `$1.50/1M`) and output (`$1.50/1M` vs. `$9.00/1M`) than `gemini-3.5-flash`, and bills **0 thought tokens** in sub-agent calls.
3. **Set `MY_THINKING = "LOW"`**: Keeps Coordinator reasoning overhead minimal (`4,732` thought tokens across all 16 holdout queries).
4. **Set `MY_TOPK = 1`**: Since each corpus document is a self-contained runbook entry or accepted Stack Overflow answer, `top_k = 1` retrieves the exact target match while minimizing sub-agent input tokens and Coordinator synthesis tokens.

### 8.3 Final Challenge Code (`Part 6`)
```python
# ===================================================================================================================
# [TODO: YOUR OPTIMIZATION (fill in every TODO below; the assert stops the cell until you do)]
MY_MODELS = {"coordinator": "gemini-3.5-flash",
             "docs":        "gemini-3.1-flash-lite",
             "community":   "gemini-3.1-flash-lite",
             "embedding":   "gemini-embedding-001"}
MY_THINKING = "LOW"     # one word for the whole agent
MY_TOPK     = 1         # one number, allowed range 1 to 5
MY_MAXOUT   = None      # leave as None
# ====================================================================================================================
```

### 8.4 Holdout Evaluation Output (16 Held-Out Queries)
- **Cost per query**: **`$0.01104395`** (`$0.0110` $\le \$0.0135$ budget)
- **Quality score**: **`0.9062`** (`0.91` $\ge 0.90$ quality floor)
- **Verdict**: **`PASS`**

**Per-Agent Spend Breakdown Across All 16 Holdout Queries:**

| Agent | Model | Input Tokens | Thought Tokens | Output Tokens | Total Cost (16 Queries) | Cost / Query |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`CommunityAgent`** | `gemini-3.1-flash-lite` | `2,723` | `0` | `4,547` | `$0.00751670` | `$0.000470` |
| **`DocsAgent`** | `gemini-3.1-flash-lite` | `3,006` | `0` | `2,541` | `$0.01806300` *(incl. VAIS queries)* | `$0.001129` |
| **`Coordinator`** | `gemini-3.5-flash` | `16,851` | `4,732` | `9,251` | `$0.15112350` | `$0.009445` |
| **Total** | — | **`22,580`** | **`4,732`** | **`16,339`** | **`$0.17670320`** | **`$0.011044`** |

### 8.5 Export & Upload `final_metrics.json` to Cloud Storage
Running the final tracking cell writes `/home/jupyter/final_metrics.json` and uploads it to `gs://<PROJECT_ID>-metrics-result/results/final_metrics.json`:
```json
{
  "cost_per_query": 0.01104395,
  "quality": 0.9062,
  "budget": 0.0135,
  "quality_floor": 0.9,
  "passed": true
}
```

> **Checkpoint 2 (Task 5)**: Click **Check my progress** in the lab instructions to verify **Optimize the Code Assistant Agent's cost and quality**.
