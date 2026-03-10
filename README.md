# ✍️ AI Blog Writing Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Pipeline-FF6B35?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI_Framework-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Free_Inference-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Tavily](https://img.shields.io/badge/Tavily-Web_Research-00C7B7?style=for-the-badge)

**A production-ready, multi-node LangGraph agent that autonomously researches, plans, writes, and illustrates full technical blog posts — with a live-streaming Streamlit UI.**

</div>

---

## 📖 Table of Contents

- [What It Does](#-what-it-does)
- [Architecture](#-architecture)
- [Agent Pipeline (Node-by-Node)](#-agent-pipeline-node-by-node)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [UI Overview](#-ui-overview)
- [Model & Provider Configuration](#-model--provider-configuration)
- [Web Research Mode](#-web-research-mode)
- [Image Generation](#-image-generation)
- [Error Handling & Resilience](#-error-handling--resilience)
- [Output Format](#-output-format)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 What It Does

Given a **topic** and a **date**, this agent:

1. 🔍 **Decides** whether the topic needs live web research or can be answered from training knowledge
2. 🌐 **Searches** the web via Tavily if needed, gathering real evidence
3. 🗂️ **Plans** a structured blog with 5–7 sections, audience targeting, and per-section metadata
4. ✍️ **Writes** each section independently with appropriate tone, code examples, and citations
5. 🖼️ **Generates** AI images (FLUX.1-schnell) and embeds them into the final post
6. 💾 **Saves** the complete blog as a timestamped `.md` file with resolved image paths

All of this streams **live** into a tabbed Streamlit UI — you watch each node complete in real time.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (Agent.py)                          │
│                                                                         │
│  Sidebar                     Main Panel (5 Tabs)                        │
│  ┌──────────────┐            ┌──────────┬──────────┬──────────────────┐ │
│  │ 🔑 API Keys  │            │ 🗂️ Plan  │ 🔍 Evid. │ 📄 MD Preview   │ │
│  │ ⚙️ LLM Model │            ├──────────┴──────────┴──────────────────┤ │
│  │ 🖼️ Img Model │            │ 🖼️ Images │ 📋 Logs                    │ │
│  │ 📝 Topic     │            │                                         │ │
│  │ 📅 As-of Date│            │  Live updates as each node completes ←  │ │
│  │ 🚀 Generate  │            └─────────────────────────────────────────┘ │
│  └──────────────┘                                                        │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │  agent.stream(initial_state)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     LANGGRAPH COMPILED GRAPH                            │
│                                                                         │
│   START                                                                 │
│     │                                                                   │
│     ▼                                                                   │
│  ┌──────────┐   needs_research=True    ┌────────────┐                  │
│  │  ROUTER  │ ───────────────────────► │  RESEARCH  │                  │
│  │  (Node)  │                          │   (Node)   │                  │
│  └──────────┘   needs_research=False   └─────┬──────┘                  │
│       │                                      │                          │
│       └──────────────────────────────────────┘                          │
│                                     │                                   │
│                                     ▼                                   │
│                             ┌──────────────┐                            │
│                             │ ORCHESTRATOR │                            │
│                             │   (Node)     │                            │
│                             └──────┬───────┘                            │
│                                    │                                    │
│                                    ▼                                    │
│                             ┌──────────────┐                            │
│                             │    WORKER    │                            │
│                             │   (Node)     │                            │
│                             └──────┬───────┘                            │
│                                    │                                    │
│                                    ▼                                    │
│                             ┌──────────────┐                            │
│                             │   REDUCER    │ ── generates images        │
│                             │   (Node)     │ ── assembles final MD      │
│                             └──────┬───────┘ ── saves to disk          │
│                                    │                                    │
│                                   END                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Agent Pipeline (Node-by-Node)

### 1. 🧭 Router Node
Classifies the topic into one of three research modes using the LLM:

| Mode | When Used | Research? |
|------|-----------|-----------|
| `closed_book` | Timeless topics (algorithms, math, established tech) | ❌ No |
| `hybrid` | Known concept + some recent context needed | ✅ Yes (targeted) |
| `open_book` | Recent events, new releases, rankings | ✅ Yes (broad) |

```
Input:  topic + as_of_date
Output: { mode, needs_research: bool, queries: ["...", ...] }
```

---

### 2. 🌐 Research Node *(conditional — only if needs_research=True)*
Fires all search queries in parallel via the **Tavily API** and deduplicates results by URL.

```
Input:  queries[]
Output: evidence[] → [{ title, url, snippet, published_at, source }]
        (max 5 results per query, deduplicated across all queries)
```

---

### 3. 🗂️ Orchestrator Node
Uses the LLM to produce a **structured blog plan** (validated via Pydantic):

```python
Plan:
  blog_title    → str
  audience      → str   (e.g. "Software engineers")
  tone          → str   (e.g. "Informative and approachable")
  blog_kind     → explainer | tutorial | news_roundup | comparison | system_design
  constraints   → ["Use concrete examples", ...]
  tasks[]       → 5–7 Task objects, each with:
      id, title, goal, bullets[], target_words (150–500),
      requires_research, requires_citations, requires_code
```

Falls back to a sensible default 5-section plan if LLM JSON parsing fails.

---

### 4. ✍️ Worker Node
Writes **each section independently** in a single LLM call, respecting:
- All bullets from the plan, in order
- ±15% word count tolerance
- Code blocks when `requires_code=True`
- Evidence snippets when `requires_research=True`
- Starts each section with `## <Section Title>`

```
Input:  tasks[], plan metadata, evidence[]
Output: sections[] → list of Markdown strings, one per task
```

---

### 5. 🎨 Reducer Node
Assembles the final blog. Does 3 things:

```
① Plans image placement  → calls LLM for 2 ImageSpec objects
                            { placeholder, filename, alt, caption,
                              prompt, after_section, size }

② Generates images       → calls HuggingFace InferenceClient
                            (FLUX.1-schnell via fal-ai / wavespeed / auto)
                            Saves PNG to ./blog_images/

③ Assembles final MD     → header + sections + image embeds + References
                            Saves timestamped .md to disk
```

---

## 📁 Project Structure

```
AI-Agent-Blog-writer/
│
├── Agent.py                  ← Entire app: LangGraph agent + Streamlit UI
│
├── blog_images/              ← Auto-created; stores AI-generated PNGs
│   ├── topic_diagram.png
│   └── topic_overview.png
│
├── <topic>_<YYYYMMDD_HHMM>.md   ← Generated blog posts (timestamped)
│
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/pratiksinha312511/AI-Agent-Blog-writer.git
cd AI-Agent-Blog-writer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Get API Keys

| Key | Where to Get | Required? |
|-----|-------------|-----------|
| HuggingFace Token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | ✅ Yes |
| Tavily API Key | [tavily.com](https://tavily.com) | ⚙️ Only for web research |

> ⚠️ **HuggingFace token must have** `"Make calls to Inference Providers"` **permission enabled.**

### 3. Run

```bash
streamlit run Agent.py
```

Open browser → `http://localhost:8501`

### 4. Generate a Blog

1. Paste your HuggingFace token in the sidebar
2. (Optional) Add Tavily key and enable **"Web research"**
3. Type your topic (e.g. `"Self Attention in Transformers"`)
4. Set the As-of date
5. Click **🚀 Generate Blog**
6. Watch tabs populate live as each node completes

---

## 🖥️ UI Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Sidebar                  │  Main Panel                          │
│ ─────────────────────     │ ──────────────────────────────────   │
│  🔑 HuggingFace Token     │  [🗂️ Plan] [🔍 Evidence] [📄 Preview]│
│  🔑 Tavily Key            │  [🖼️ Images] [📋 Logs]              │
│  ☑ Enable web research    │                                      │
│                           │  🗂️ Plan Tab:                        │
│  ⚙️ LLM Model dropdown    │    Blog title, audience, tone        │
│  ⚙️ LLM Provider dropdown │    Table: id | title | words | flags │
│  🖼️ Image Model dropdown  │                                      │
│                           │  🔍 Evidence Tab:                    │
│  📝 Topic textarea        │    Cards: title, URL, snippet        │
│  📅 As-of date input      │                                      │
│                           │  📄 Preview Tab:                     │
│  🚀 Generate Blog btn     │    Full rendered Markdown blog       │
│                           │    (images embedded as base64)       │
│  📚 Past Blogs list       │                                      │
│    🔴 Most recent         │  🖼️ Images Tab:                      │
│    ⚪ Older entries        │    Generated PNGs in 2-col grid      │
│                           │                                      │
│                           │  📋 Logs Tab:                        │
│                           │    Live monospace log stream         │
│                           │    (node name + per-node metrics)    │
└──────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Model & Provider Configuration

### Text (LLM) Models — All Free on HF Free Tier

| Model | ID | Best For |
|-------|----|----------|
| **Llama 3.1 8B** *(recommended)* | `meta-llama/Meta-Llama-3.1-8B-Instruct` | General writing |
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B-Instruct` | Technical content |
| Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.3` | Instruction following |

### LLM Providers — Free Quota Only

| Provider | Speed | Notes |
|----------|-------|-------|
| `auto` ✅ | Variable | HF picks best available free provider |
| `cerebras` ✅ | ⚡ Fastest (2000+ tok/s) | Best for large models |
| `sambanova` ✅ | Fast | Good 70B support |
| `novita` ✅ | Moderate | Wide model support |
| `nscale` ✅ | Reliable | Consistent uptime |

> ❌ `groq`, `together`, `fireworks-ai` removed — return 402 on free accounts.

### Image Models — Free via fal-ai / wavespeed

| Model | ID | Speed |
|-------|----|-------|
| **FLUX.1-schnell** *(recommended)* | `black-forest-labs/FLUX.1-schnell` | ⚡ 1–4 steps |
| Stable Diffusion XL | `stabilityai/stable-diffusion-xl-base-1.0` | Moderate |

---

## 🌐 Web Research Mode

When **"Enable web research"** is checked and a Tavily key is provided:

```
Router → detects topic needs fresh data
       → generates 3–5 targeted search queries

Research node → fires all queries against Tavily API
              → deduplicates by URL
              → returns up to 15 evidence items

Orchestrator → uses top 6 snippets to inform the plan
Worker       → injects relevant snippets into sections
               that have requires_research=True
Reducer      → appends a ## References section with
               numbered citations [1]...[N]
```

**Without Tavily:** Agent runs in `closed_book` mode — no research node is invoked.

---

## 🖼️ Image Generation

The Reducer node plans and generates 2 images per blog:

```
① LLM plans image specs:
   - prompt   → "Clean technical diagram of X, white background, minimal"
   - filename → "topic_diagram.png"
   - caption  → "Figure 1: ..."
   - after_section → int (where to insert in blog)

② InferenceClient generates image:
   Primary provider:   auto
   Fallbacks:          wavespeed → fal-ai
   Saves to:           ./blog_images/<filename>.png

③ Markdown Preview:
   Local paths → embedded as base64 data URIs
   (st.markdown() cannot load local file paths directly)

④ Saved .md file:
   ./blog_images/ → replaced with absolute path
   so the file renders in any Markdown viewer
```

---

## 🛡️ Error Handling & Resilience

The agent handles all common API failure modes gracefully:

| Error | Behaviour |
|-------|-----------|
| `402 Payment Required` | Immediate fail with fix instructions (wrong provider) |
| `401 Unauthorized` | Immediate fail with token permission instructions |
| `503 Model Loading` | Auto-retry after 40s (up to 3 attempts) |
| `429 Rate Limited` | Auto-retry after 60s |
| LLM JSON parse failure | Falls back to hardcoded 5-section default plan |
| Image generation failure | Inserts HTML comment placeholder, blog still saves |
| All image providers fail | Logs clearly, continues without images |

---

## 📄 Output Format

Generated blog is saved as:

```
<topic_slug>_<YYYYMMDD_HHMM>.md
```

Structure:

```markdown
# Blog Title

*Published: 2026/01/30 | Audience: ... | Tone: ...*

---

## Section 1

<content>

![alt text](./blog_images/diagram.png)
*Figure 1: Caption*

## Section 2

<content with code blocks if requires_code=True>

...

## References

[1] [Article Title](https://url.com) (2026-01-15)
[2] ...
```

---

## 🔧 Troubleshooting

**Agent won't start / import errors**
```bash
pip install streamlit langgraph langchain huggingface_hub pydantic requests
```

**402 Payment Required on LLM calls**
→ Change the Provider dropdown to `auto` in the sidebar.

**401 Unauthorized on HuggingFace**
→ Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → edit your token → enable **"Make calls to Inference Providers"**.

**Images not generating**
→ Same HF token permission issue. Check logs tab for provider-specific error messages.

**Blog is slow to generate**
→ Normal — 7B LLMs via free HF inference take 1–3 min total. The Logs tab shows live progress.

**Streamlit page goes blank mid-run**
→ Streamlit reruns on state change. This is expected — tabs repopulate automatically.

---

## 📦 Dependencies

```txt
streamlit
langgraph
langchain
langchain-community
huggingface_hub
pydantic
requests
```

---

## 📜 License

This project is open for learning and demonstration. Fork freely, extend as needed.

---

<div align="center">

Built with ❤️ using **LangGraph** · **Streamlit** · **HuggingFace** · **Tavily**

</div>
