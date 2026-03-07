"""
Blog Writing Agent — Streamlit UI
Mirrors the demo UI from the screenshots:
  - Left sidebar: Topic input, As-of date, Generate button, Past blogs list
  - Right panel: Plan | Evidence | Markdown Preview | Images | Logs tabs
  - Live streaming: each node updates the UI as it completes
"""

import os, json, re, time, threading, queue
import requests                       # used for Tavily + direct HF image API
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Literal, Optional, TypedDict

import streamlit as st
from pydantic import BaseModel, Field

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS matching screenshots ──────────────────────────────────────
st.markdown("""
<style>
/* Overall dark background */
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }

/* Sidebar title */
.sidebar-title { font-size: 1.1rem; font-weight: 700; color: #e6edf3; margin-bottom: 1rem; }

/* Tab styling */
[data-testid="stTabs"] [role="tab"] {
    color: #8b949e; font-size: 0.85rem; padding: 6px 14px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #58a6ff; border-bottom: 2px solid #58a6ff;
}

/* Plan table */
.plan-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.plan-table th { background: #21262d; color: #8b949e; padding: 6px 10px; text-align: left; }
.plan-table td { padding: 6px 10px; border-bottom: 1px solid #21262d; color: #e6edf3; }
.plan-table tr:hover td { background: #161b22; }

/* Log area */
.log-box {
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    padding: 12px; font-family: monospace; font-size: 0.78rem;
    color: #7ee787; height: 340px; overflow-y: auto; white-space: pre-wrap;
}

/* Node badge */
.node-badge {
    display: inline-block; background: #388bfd26; color: #58a6ff;
    border: 1px solid #388bfd; border-radius: 4px;
    padding: 1px 8px; font-size: 0.75rem; font-family: monospace;
}

/* Generate button */
[data-testid="stButton"] > button {
    background: #da3633; color: white; border: none;
    border-radius: 6px; font-weight: 600; width: 100%;
}
[data-testid="stButton"] > button:hover { background: #b91c1c; }

/* Past blog items */
.past-blog { font-size: 0.78rem; color: #8b949e; cursor: pointer;
             padding: 4px 0; border-bottom: 1px solid #21262d; }
.past-blog:hover { color: #e6edf3; }

/* Evidence card */
.ev-card { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
           padding: 10px 14px; margin-bottom: 8px; }
.ev-title { color: #58a6ff; font-size: 0.85rem; font-weight: 600; }
.ev-url { color: #8b949e; font-size: 0.75rem; }
.ev-snippet { color: #c9d1d9; font-size: 0.8rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "running": False,
        "logs": [],
        "plan": None,
        "tasks": [],
        "evidence": [],
        "sections": [],
        "image_specs": [],
        "images": [],
        "final_blog": "",
        "mode": "",
        "needs_research": False,
        "queries": [],
        "past_blogs": [],          # list of {title, path, timestamp}
        "current_node": "",
        "error": "",
        "done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════
# AGENT IMPORTS — load lazily so the UI still renders if not installed
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_agent(hf_key: str, tavily_key: str, use_research: bool,
               llm_provider: str, text_model: str, image_model: str):
    """Build and cache the LangGraph blog agent."""
    import requests as _requests
    from huggingface_hub import InferenceClient
    from langgraph.graph import StateGraph, START, END

    # ── Data models ──────────────────────────────────────────────
    class RouterDecision(BaseModel):
        needs_research: bool
        mode: Literal["closed_book", "hybrid", "open_book"]
        queries: List[str] = Field(default_factory=list)

    class Task(BaseModel):
        id: int
        title: str
        goal: str
        bullets: List[str] = Field(default_factory=list)
        target_words: int = 300
        tags: List[str] = Field(default_factory=list)
        requires_research: bool = False
        requires_citations: bool = False
        requires_code: bool = False

    class Plan(BaseModel):
        blog_title: str
        audience: str
        tone: str
        blog_kind: Literal["explainer","tutorial","news_roundup","comparison","system_design"] = "explainer"
        constraints: List[str] = Field(default_factory=list)
        tasks: List[Task] = Field(default_factory=list)

    class ImageSpec(BaseModel):
        placeholder: str
        filename: str
        alt: str
        caption: str
        prompt: str
        after_section: int = 1
        size: str = "1024x1024"

    class BlogState(TypedDict):
        topic: str; as_of_date: str; mode: str; needs_research: bool
        queries: List[str]; evidence: List[Dict]; evidence_count: int
        plan: Optional[Dict]; tasks: Optional[List[Dict]]
        sections: List[str]; sections_done: int
        image_specs: List[Dict]; images: List[str]; final_blog: str

    IMAGES_DIR = Path("./blog_images")
    IMAGES_DIR.mkdir(exist_ok=True)

    # ── call_llm ─────────────────────────────────────────────────
    def call_llm(system_prompt, user_prompt, max_tokens=512, temperature=0.3):
        client = InferenceClient(provider="auto", api_key=hf_key)
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=text_model,
                    messages=[{"role":"system","content":system_prompt},
                               {"role":"user","content":user_prompt}],
                    max_tokens=max_tokens, temperature=temperature,
                )
                return r.choices[0].message.content.strip()
            except Exception as e:
                s = str(e).lower()
                # 402 = paid provider on free account — fail immediately with clear message
                if "402" in s or "payment required" in s:
                    raise RuntimeError(
                        f"402 Payment Required — provider '{llm_provider}' requires a paid HF account.\n"
                        f"👉 Fix: Change the Provider dropdown to 'auto' in the sidebar."
                    )
                elif "401" in s or "unauthorized" in s:
                    raise RuntimeError(
                        f"401 Unauthorized — your HF token is invalid or missing permissions.\n"
                        f"👉 Fix: Go to huggingface.co/settings/tokens → enable 'Make calls to Inference Providers'."
                    )
                elif "503" in s or "loading" in s: time.sleep(40)
                elif "429" in s or "rate" in s:    time.sleep(60)
                elif attempt < 2:                  time.sleep(5)
                else:
                    raise RuntimeError(f"LLM failed: {e}")
        raise RuntimeError("LLM failed after 3 attempts")

    # ── extract_json ─────────────────────────────────────────────
    def extract_json(text):
        for pat in [r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```",
                    r"(\{[\s\S]*\})", r"(\[[\s\S]*\])"]:
            m = re.search(pat, text, re.DOTALL)
            if m:
                try: return json.loads(m.group(1).strip())
                except: continue
        return json.loads(text.strip())

    # ── tavily_search ────────────────────────────────────────────
    def tavily_search(query, max_results=5):
        if not use_research: return []
        try:
            r = _requests.post("https://api.tavily.com/search",
                json={"api_key":tavily_key,"query":query,
                      "max_results":max_results,"search_depth":"basic"}, timeout=20)
            r.raise_for_status()
            return [{"title":x.get("title",""),"url":x.get("url",""),
                     "published_at":x.get("published_date"),
                     "snippet":x.get("content","")[:500],"source":x.get("source","")}
                    for x in r.json().get("results",[])]
        except: return []

    # generate_image is defined OUTSIDE load_agent (see below)
    # so it is never frozen by @st.cache_resource and always
    # uses the current hf_key / image_model / add_log.

    # ── fallback plan ────────────────────────────────────────────
    def _fallback_plan(topic):
        return Plan(
            blog_title=f"Understanding {topic}: A Complete Guide",
            audience="Software engineers and technical practitioners",
            tone="Informative and approachable", blog_kind="explainer",
            constraints=["Use concrete examples","Keep under 2000 words"],
            tasks=[
                Task(id=1,title=f"Introduction to {topic}",goal="Set context",
                     bullets=["Definition","Why it matters","Overview"],target_words=200),
                Task(id=2,title="Core Concepts",goal="Explain fundamentals",
                     bullets=["Key components","How it works","Walkthrough"],target_words=350),
                Task(id=3,title="Code Example",goal="Concrete implementation",
                     bullets=["Setup","Code walkthrough","Output"],target_words=400,requires_code=True),
                Task(id=4,title="Best Practices",goal="Expert guidance",
                     bullets=["When to use","Pitfalls","Performance"],target_words=250),
                Task(id=5,title="Conclusion",goal="Wrap up",
                     bullets=["Summary","Next steps"],target_words=150),
            ])

    # ── ROUTER_SYSTEM ────────────────────────────────────────────
    ROUTER_SYSTEM = """You are a blog research routing expert.
Analyze the topic and output research strategy as JSON.
Rules:
- Timeless topics (algorithms, math, established tech) → needs_research=false, mode=closed_book
- Recent events, new releases, rankings → needs_research=true, mode=open_book
- Known concept + some recent context → needs_research=true, mode=hybrid
Output ONLY valid JSON. No markdown. No explanation.
Schema: {"needs_research": bool, "mode": "closed_book|hybrid|open_book", "queries": ["...", ...]}"""

    ORCHESTRATOR_SYSTEM = """You are a senior technical content strategist.
Output a blog plan as a JSON object with these fields:
  blog_title (str), audience (str), tone (str),
  blog_kind (explainer|tutorial|news_roundup|comparison|system_design),
  constraints (list of str), tasks (list of Task).
Each Task: id (int), title (str), goal (str), bullets (list of str),
target_words (int 150-500), tags (list), requires_research (bool),
requires_citations (bool), requires_code (bool).
Plan 5-7 sections. Output ONLY valid JSON. No markdown. No explanation."""

    WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE blog section in Markdown.
Hard rules:
- Cover ALL bullets provided, in order.
- Stay within ±15% of target word count.
- Start with '## <Section Title>'.
- Output ONLY the section. No H1 title. No preamble.
Quality bar:
- Be precise. Include code when requires_code=true.
- Explain trade-offs. Call out edge cases. No fluff."""

    IMAGE_PLANNER_SYSTEM = """You are planning images for a technical blog.
Output ONLY a JSON array of 2 image specs. No markdown. No explanation.
Each spec: placeholder (str), filename (str), alt (str), caption (str),
prompt (str — technical diagram, white background, minimal style),
after_section (int — 0-based index), size ("1024x1024")."""

    # ── Nodes ────────────────────────────────────────────────────
    def router_node(state):
        raw = call_llm(ROUTER_SYSTEM,
            f"Topic: {state['topic']}\nAs-of date: {state.get('as_of_date','2026/01/30')}\n"
            f"Output JSON: needs_research (bool), mode, queries (3-5 strings if needs_research else []).",
            max_tokens=512, temperature=0.1)
        try:
            d = RouterDecision(**extract_json(raw))
        except:
            d = RouterDecision(needs_research=False, mode="closed_book", queries=[])
        if not use_research:
            d = RouterDecision(needs_research=False, mode="closed_book", queries=[])
        return {**state, "mode":d.mode, "needs_research":d.needs_research,
                "queries":d.queries, "evidence":[], "evidence_count":0,
                "plan":None, "tasks":None, "sections":[], "sections_done":0,
                "image_specs":[], "images":[], "final_blog":""}

    def research_node(state):
        all_ev, seen = [], set()
        for q in state.get("queries",[]):
            for r in tavily_search(q, max_results=3):
                u = r.get("url","")
                if u and u not in seen:
                    seen.add(u); all_ev.append(r)
        return {**state, "evidence":all_ev, "evidence_count":len(all_ev)}

    def orchestrator_node(state):
        ev = state.get("evidence",[])
        ev_sum = ("Web research:\n" +
            "\n".join(f"[{i+1}] {e.get('title','')}: {e.get('snippet','')[:120]}"
                      for i,e in enumerate(ev[:6]))
        ) if ev else "No web research. Use training knowledge."
        raw = call_llm(ORCHESTRATOR_SYSTEM,
            f"Topic: {state['topic']}\nMode: {state.get('mode','closed_book')}\n"
            f"Evidence count: {len(ev)}\n{ev_sum}\n\nCreate the blog plan.",
            max_tokens=1500, temperature=0.4)
        try:
            plan = Plan(**extract_json(raw))
        except:
            plan = _fallback_plan(state["topic"])
        return {**state, "plan":plan.model_dump(), "tasks":[t.model_dump() for t in plan.tasks]}

    def worker_node(state):
        tasks, plan, ev = state.get("tasks",[]), state.get("plan",{}), state.get("evidence",[])
        sections = []
        for task in tasks:
            bullets = "\n".join(f"- {b}" for b in task.get("bullets",[]))
            ev_ctx  = ""
            if task.get("requires_research") and ev:
                ev_ctx = "\nEvidence:\n" + "\n".join(
                    f"[{i+1}] {e.get('title','')}: {e.get('snippet','')[:150]}"
                    for i,e in enumerate(ev[:4]))
            md = call_llm(WORKER_SYSTEM,
                f"Section: {task['title']}\nGoal: {task.get('goal','')}\n"
                f"Bullets:\n{bullets}\nTarget words: {task.get('target_words',300)}\n"
                f"Requires code: {task.get('requires_code',False)}\n\n"
                f"Blog: {plan.get('blog_title','')} | Audience: {plan.get('audience','')} | Tone: {plan.get('tone','')}"
                f"{ev_ctx}", max_tokens=900, temperature=0.5)
            if not md.strip().startswith("##"):
                md = f"## {task['title']}\n\n{md}"
            sections.append(md)
        return {**state, "sections":sections, "sections_done":len(sections)}

    def _plan_images(sections, plan, topic):
        titles = []
        for s in sections:
            for line in s.split("\n"):
                if line.startswith("##"):
                    titles.append(line.lstrip("#").strip()); break
        raw = call_llm(IMAGE_PLANNER_SYSTEM,
            f"Blog: {plan.get('blog_title',topic)}\nTopic: {topic}\n"
            f"Sections: {', '.join(titles)}\nPlan 2 images.",
            max_tokens=500, temperature=0.2)
        try:
            m = re.search(r"(\[[\s\S]*\])", raw)
            return json.loads(m.group(1))[:3] if m else []
        except:
            safe = topic.lower().replace(" ","_")
            return [{"placeholder":"[[IMAGE_1]]","filename":f"{safe}_diagram.png",
                     "alt":f"Diagram of {topic}","caption":f"Figure 1: {topic} overview",
                     "prompt":f"Clean technical diagram of {topic}, white background, minimal",
                     "after_section":1,"size":"1024x1024"}]

    def reducer_node(state):
        sections, plan, ev = state.get("sections",[]), state.get("plan",{}), state.get("evidence",[])
        topic, as_of = state["topic"], state.get("as_of_date","")
        img_specs = _plan_images(sections, plan, topic)
        generated = []
        for spec in img_specs:
            fp = generate_image_standalone(
                prompt    = spec.get("prompt", f"Clean technical diagram of {topic}, white background, minimal style"),
                filename  = spec.get("filename", f"img_{len(generated)+1}.png"),
                hf_key    = hf_key,
                image_model = image_model,
            )
            if fp: spec["generated_path"] = fp; generated.append(fp)
        secs = list(sections)
        for spec in sorted(img_specs, key=lambda x: x.get("after_section",0), reverse=True):
            idx = min(max(spec.get("after_section",0),0), len(secs)-1)
            if spec.get("generated_path"):
                fname  = os.path.basename(spec["generated_path"])
                img_md = f"\n\n![{spec.get('alt','')}](./blog_images/{fname})\n*{spec.get('caption','')}*\n"
            else:
                img_md = f"\n\n<!-- {spec.get('placeholder','')} — image not generated -->\n"
            secs.insert(idx+1, img_md)
        refs = ""
        if ev:
            lines = ["\n\n## References\n"]
            for i,e in enumerate(ev[:10]):
                pub = f" ({e['published_at']})" if e.get("published_at") else ""
                lines.append(f"[{i+1}] [{e.get('title','Source')}]({e.get('url','')}){pub}")
            refs = "\n".join(lines)
        header = (f"# {plan.get('blog_title',topic)}\n\n"
                  f"*Published: {as_of} | Audience: {plan.get('audience','')} | Tone: {plan.get('tone','')}*\n\n---\n\n")
        final_blog = header + "\n\n".join(secs) + refs
        return {**state, "image_specs":img_specs, "images":generated, "final_blog":final_blog}

    def should_research(state):
        return "research" if state.get("needs_research") and state.get("queries") else "orchestrator"

    # ── Build graph ──────────────────────────────────────────────
    g = StateGraph(BlogState)
    g.add_node("router",       router_node)
    g.add_node("research",     research_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("worker",       worker_node)
    g.add_node("reducer",      reducer_node)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", should_research,
                             {"research":"research","orchestrator":"orchestrator"})
    g.add_edge("research","orchestrator")
    g.add_edge("orchestrator","worker")
    g.add_edge("worker","reducer")
    g.add_edge("reducer", END)
    return g.compile(), BlogState


# ══════════════════════════════════════════════════════════════════
# IMAGE GENERATION — standalone, NOT inside @st.cache_resource
# so it always uses the live hf_key and can write to session logs.
#
# ROOT CAUSE OF BUG: The old code used `provider=llm_provider` which
# maps to e.g. "groq" — Groq only serves text models and returns 404
# on image endpoints. Images need provider="auto" which routes to
# fal-ai or hf-inference, both of which support FLUX.1-schnell.
# ══════════════════════════════════════════════════════════════════
_IMAGES_DIR = Path("./blog_images")
_IMAGES_DIR.mkdir(exist_ok=True)

# Module-level log sink — set by run_agent before each run so that
# generate_image_standalone can write to the Streamlit session log.
_log_sink = print   # default: print to terminal

def generate_image_standalone(prompt: str, filename: str,
                               hf_key: str, image_model: str,
                               _log=None) -> Optional[str]:
    """
    Generate an image via HuggingFace InferenceClient.
    Uses fal-ai as primary provider (free, fast, confirmed working for FLUX),
    falls back to wavespeed, then auto.
    Both fal-ai and wavespeed are free on HF free tier.
    Source: huggingface.co/docs/inference-providers/providers/wavespeed
    """
    import traceback
    from huggingface_hub import InferenceClient as _IC
    if _log is None:
        _log = _log_sink

    # Free image providers in priority order (all confirmed free on HF free tier)
    # fal-ai   → primary, fast, confirmed free for FLUX.1-schnell & FLUX.1-dev
    # wavespeed → secondary, very fast, free for FLUX models
    # auto     → last resort, HF picks any available provider
    image_providers = ["auto"]

    for provider in image_providers:
        _log(f"     🌐 Trying image provider: {provider}")
        for attempt in range(2):
            try:
                _log(f"     🎨 Generating '{filename}' (attempt {attempt+1}/2)...")
                client = _IC(provider=provider, api_key=hf_key)
                img = client.text_to_image(prompt=prompt, model=image_model)
                fp  = _IMAGES_DIR / filename
                img.save(str(fp))
                _log(f"     ✅ Saved via {provider}: {fp} ({fp.stat().st_size // 1024} KB)")
                return str(fp)
            except Exception as e:
                err = str(e)
                _log(f"     ⚠️  {provider} attempt {attempt+1} failed: {err[:120]}")
                if "401" in err or "unauthorized" in err.lower():
                    _log("     ❌ 401 — check HF token has 'Make calls to Inference Providers' permission")
                    return None   # Bad token — no point retrying any provider
                if "402" in err or "payment" in err.lower():
                    _log(f"     ⚠️  {provider} requires payment — trying next provider...")
                    break         # Skip to next provider
                if "503" in err or "loading" in err.lower():
                    _log("     ⏳ Model loading, waiting 30s...")
                    time.sleep(30)
                elif "429" in err or "rate" in err.lower():
                    _log("     ⏳ Rate limited, waiting 60s...")
                    time.sleep(60)
                elif attempt < 1:
                    time.sleep(5)

    _log(f"     ❌ All image providers failed for: {filename}")
    return None


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-title">✍️ Blog Writing Agent</div>', unsafe_allow_html=True)

    st.markdown("**🔑 API Keys**")
    hf_key = st.text_input("HuggingFace Token", type="password",
                            value=os.environ.get("HF_API_KEY",""),
                            placeholder="hf_...")
    tavily_key = st.text_input("Tavily Key (optional)", type="password",
                                value=os.environ.get("TAVILY_API_KEY",""),
                                placeholder="tvly_...")
    use_research = st.checkbox("Enable web research (needs Tavily)", value=False)

    st.divider()
    st.markdown("**⚙️ Model**")

    # ── FREE TEXT MODELS (confirmed working on HF free tier) ─────
    # Routed via provider="auto" or specific free providers below.
    # Source: huggingface.co/docs/inference-providers + cerebras docs
    FREE_TEXT_MODELS = {
    "Llama 3.1 8B (recommended)": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral 7B": "mistralai/Mistral-7B-Instruct-v0.3",
    }
    text_model = FREE_TEXT_MODELS[st.selectbox(
        "LLM (Text Model)",
        list(FREE_TEXT_MODELS.keys()),
        index=0,
        help="All models are free on HF free-tier. Llama 3.3 70B recommended.",
    )]

    # ── FREE LLM PROVIDERS (no 402, confirmed free quota) ────────
    # cerebras  → fastest (2000+ tok/s), free, best for 70B
    # sambanova → fast, free quota, great 70B support
    # novita    → free quota, wide model support
    # nscale    → free quota, reliable
    # auto      → HF picks best available free provider automatically
    # REMOVED:  groq / together / fireworks-ai → 402 on free accounts
    FREE_LLM_PROVIDERS = {
        "auto      ✅ HF picks best free provider": "auto",
        "cerebras  ✅ fastest (2000+ tok/s), free":  "cerebras",
        "sambanova ✅ fast 70B, free":               "sambanova",
        "novita    ✅ wide support, free":            "novita",
        "nscale    ✅ reliable, free":                "nscale",
    }
    llm_provider = FREE_LLM_PROVIDERS[st.selectbox(
        "LLM Provider",
        list(FREE_LLM_PROVIDERS.keys()),
        index=0,
        help="Only free-tier providers. groq/together/fireworks-ai removed — 402 on free accounts.",
    )]

    # ── FREE IMAGE MODELS (confirmed free via fal-ai / wavespeed) ─
    # FLUX.1-schnell → fastest (1-4 steps), free via fal-ai & wavespeed
    # FLUX.1-dev     → higher quality, free via fal-ai & wavespeed
    # Source: huggingface.co/docs/inference-providers/providers/wavespeed
    FREE_IMAGE_MODELS = {
    "FLUX Schnell (fast)": "black-forest-labs/FLUX.1-schnell",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    }
    image_model_selected = FREE_IMAGE_MODELS[st.selectbox(
        "Image Model",
        list(FREE_IMAGE_MODELS.keys()),
        index=0,
        help="Both models are free via fal-ai / wavespeed providers. schnell is faster.",
    )]

    st.divider()
    st.markdown("**📝 Generate New Blog**")
    topic    = st.text_area("Topic", value="Self Attention", height=80)
    as_of    = st.text_input("As-of date", value="2026/01/30")

    generate_btn = st.button("🚀 Generate Blog", disabled=st.session_state.running)

    # Past blogs list
    if st.session_state.past_blogs:
        st.divider()
        st.markdown("**📚 Past Blogs**")
        for i, b in enumerate(reversed(st.session_state.past_blogs[-10:])):
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown("🔴" if i == 0 else "⚪")
            with col2:
                st.markdown(f'<div class="past-blog">{b["title"][:40]}<br>'
                            f'<span style="color:#444">{b["path"]}</span></div>',
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════
st.markdown("## Blog Writing Agent")

tab_plan, tab_ev, tab_preview, tab_images, tab_logs = st.tabs(
    ["🗂️ Plan", "🔍 Evidence", "📄 Markdown Preview", "🖼️ Images", "📋 Logs"]
)

# ── Placeholders inside each tab ─────────────────────────────────
with tab_plan:
    plan_status   = st.empty()
    plan_content  = st.empty()

with tab_ev:
    ev_status  = st.empty()
    ev_content = st.empty()

with tab_preview:
    prev_status  = st.empty()
    prev_content = st.empty()

with tab_images:
    img_status  = st.empty()
    img_content = st.empty()

with tab_logs:
    log_status  = st.empty()
    log_box     = st.empty()


# ══════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════════
def render_plan():
    plan  = st.session_state.plan
    tasks = st.session_state.tasks
    if not plan:
        plan_status.info("⏳ Waiting for orchestrator to create plan...")
        return
    plan_status.empty()
    with plan_content.container():
        st.markdown(f"### {plan.get('blog_title','')}")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Audience:** {plan.get('audience','')}")
        c2.markdown(f"**Tone:** {plan.get('tone','')}")
        c3.markdown(f"**Blog kind:** {plan.get('blog_kind','')}")
        if tasks:
            rows = ""
            for t in tasks:
                res  = "☑" if t.get("requires_research") else "○"
                cite = "☑" if t.get("requires_citations") else "○"
                code = "☑" if t.get("requires_code") else "○"
                rows += (f"<tr><td>{t['id']}</td><td>{t['title']}</td>"
                         f"<td>{t.get('target_words',0)}</td>"
                         f"<td>{res}</td><td>{cite}</td><td>{code}</td></tr>")
            st.markdown(
                f"""<table class="plan-table">
                <tr><th>id</th><th>title</th><th>target_words</th>
                <th>requires_research</th><th>requires_citations</th><th>requires_code</th></tr>
                {rows}</table>""",
                unsafe_allow_html=True)

def render_evidence():
    ev = st.session_state.evidence
    if not ev:
        ev_status.info("⏳ No evidence yet. Enable research or wait for research node.")
        return
    ev_status.success(f"✅ {len(ev)} evidence items gathered")
    cards = ""
    for i, e in enumerate(ev):
        cards += (f'<div class="ev-card">'
                  f'<div class="ev-title">[{i+1}] {e.get("title","")}</div>'
                  f'<div class="ev-url"><a href="{e.get("url","")}" style="color:#58a6ff">'
                  f'{e.get("url","")[:80]}</a></div>'
                  f'<div class="ev-snippet">{e.get("snippet","")[:200]}</div>'
                  f'</div>')
    ev_content.markdown(cards, unsafe_allow_html=True)

def _embed_images_as_base64(markdown_text: str) -> str:
    """Replace local ./blog_images/... paths with base64 data URIs.
    st.markdown() cannot load local file paths — only http:// or data: URIs work."""
    import base64 as _b64, re as _re
    def _replace(m):
        alt, path = m.group(1), m.group(2)
        if path.startswith("http"):
            return m.group(0)
        fp = Path(path) if Path(path).exists() else Path(".") / path.lstrip("./")
        if fp.exists():
            try:
                mime = {"png":"png","jpg":"jpeg","jpeg":"jpeg","gif":"gif","webp":"webp"}.get(fp.suffix.lower().lstrip("."), "png")
                b64  = _b64.b64encode(fp.read_bytes()).decode()
                return f"![{alt}](data:image/{mime};base64,{b64})"
            except Exception:
                pass
        return m.group(0)
    return _re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _replace, markdown_text)


def render_preview():
    blog = st.session_state.final_blog
    if not blog:
        prev_status.info("⏳ Blog not yet generated. Run the agent first.")
        return
    prev_status.empty()
    # Embed local images as base64 — st.markdown() cannot load local file paths
    prev_content.markdown(_embed_images_as_base64(blog), unsafe_allow_html=True)

def render_images():
    images = st.session_state.images
    specs  = st.session_state.image_specs
    if not specs:
        img_status.info("⏳ No images yet.")
        return
    img_status.success(f"✅ {len(images)}/{len(specs)} images generated")
    cols = img_content.columns(min(len(images), 2)) if images else []
    for i, fp in enumerate(images):
        try:
            cols[i % 2].image(fp, caption=specs[i].get("caption","") if i < len(specs) else "")
        except: pass

def render_logs():
    logs = st.session_state.logs
    if not logs:
        log_status.info("⏳ No logs yet.")
        return
    log_status.empty()
    log_box.markdown(
        f'<div class="log-box">{"".join(logs)}</div>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# GENERATE BUTTON HANDLER
# ══════════════════════════════════════════════════════════════════
def add_log(msg: str):
    st.session_state.logs.append(msg + "\n")

def run_agent(topic, as_of, hf_key, tavily_key, use_research,
              llm_provider, text_model, image_model="black-forest-labs/FLUX.1-schnell"):
    """Run the agent and stream updates back into session state."""
    try:
        agent, BlogState = load_agent(hf_key, tavily_key, use_research,
                                      llm_provider, text_model, image_model)
    except Exception as e:
        st.session_state.error = f"Failed to load agent: {e}"
        st.session_state.running = False
        return

    initial_state = {
        "topic": topic, "as_of_date": as_of, "mode": "closed_book",
        "needs_research": False, "queries": [], "evidence": [], "evidence_count": 0,
        "plan": None, "tasks": None, "sections": [], "sections_done": 0,
        "image_specs": [], "images": [], "final_blog": "",
    }

    add_log(f"🚀 Starting agent for topic: '{topic}'\n")

    # Wire image logs into Streamlit session state
    global _log_sink
    _log_sink = add_log

    try:
        for chunk in agent.stream(initial_state):
            for node_name, out in chunk.items():
                st.session_state.current_node = node_name
                add_log(f"\n📦 Node: {node_name}")
                add_log("─" * 40)

                if node_name == "router":
                    st.session_state.mode           = out.get("mode","")
                    st.session_state.needs_research = out.get("needs_research", False)
                    st.session_state.queries        = out.get("queries", [])
                    add_log(f"  mode           : {out.get('mode')}")
                    add_log(f"  needs_research : {out.get('needs_research')}")
                    for i, q in enumerate(out.get("queries",[])):
                        add_log(f"    {i}: {q}")

                elif node_name == "research":
                    st.session_state.evidence = out.get("evidence", [])
                    add_log(f"  evidence_count : {len(st.session_state.evidence)}")

                elif node_name == "orchestrator":
                    st.session_state.plan  = out.get("plan", {})
                    st.session_state.tasks = out.get("tasks", [])
                    add_log(f"  blog_title : {st.session_state.plan.get('blog_title','')}")
                    add_log(f"  sections   : {len(st.session_state.tasks)}")

                elif node_name == "worker":
                    st.session_state.sections = out.get("sections", [])
                    total = sum(len(s.split()) for s in st.session_state.sections)
                    add_log(f"  sections_done : {len(st.session_state.sections)}")
                    add_log(f"  total_words   : {total}")

                elif node_name == "reducer":
                    st.session_state.final_blog  = out.get("final_blog", "")
                    st.session_state.image_specs = out.get("image_specs", [])
                    st.session_state.images      = out.get("images", [])
                    add_log(f"  final_words    : {len(st.session_state.final_blog.split())}")
                    add_log(f"  images_created : {len(st.session_state.images)}")

        add_log("\n✅ Blog generation complete!")

        # Save to disk — replace relative ./blog_images/ with absolute paths
        # so the .md file renders correctly when opened in any markdown viewer
        safe = re.sub(r'[^a-z0-9]+','_', topic.lower()).strip('_')
        ts   = datetime.now().strftime("%Y%m%d_%H%M")
        path = Path(f"{safe}_{ts}.md")
        abs_img_dir   = str(_IMAGES_DIR.resolve())
        blog_for_file = st.session_state.final_blog.replace("./blog_images/", abs_img_dir + "/")
        path.write_text(blog_for_file, encoding="utf-8")
        add_log(f"💾 Saved to: {path}")

        st.session_state.past_blogs.append({
            "title": st.session_state.plan.get("blog_title", topic) if st.session_state.plan else topic,
            "path":  str(path),
            "timestamp": ts,
        })

    except Exception as e:
        st.session_state.error = str(e)
        add_log(f"\n❌ ERROR: {e}")

    finally:
        st.session_state.running = False
        st.session_state.done    = True


if generate_btn:
    if not hf_key or hf_key == "hf_PASTE_YOUR_TOKEN_HERE":
        st.error("❌ Please enter your HuggingFace API token in the sidebar.")
    else:
        # Reset state
        st.session_state.running     = True
        st.session_state.done        = False
        st.session_state.error       = ""
        st.session_state.logs        = []
        st.session_state.plan        = None
        st.session_state.tasks       = []
        st.session_state.evidence    = []
        st.session_state.sections    = []
        st.session_state.image_specs = []
        st.session_state.images      = []
        st.session_state.final_blog  = ""
        st.session_state.current_node = ""

        # Run in same thread (Streamlit reruns handle the streaming)
        run_agent(topic, as_of, hf_key, tavily_key, use_research,
                  llm_provider, text_model, image_model_selected)
        st.rerun()


# ── Show running status banner ───────────────────────────────────
if st.session_state.running:
    st.info(f"⏳ Running… current node: **{st.session_state.current_node}**")
elif st.session_state.done and not st.session_state.error:
    st.success("✅ Blog generation complete!")
elif st.session_state.error:
    st.error(f"❌ {st.session_state.error}")

# ── Render all tabs ──────────────────────────────────────────────
render_plan()
render_evidence()
render_preview()
render_images()
render_logs()
