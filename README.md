# 🧠 Nero

Nero is an experimental **conscious-agent** built on a hybrid architecture: a real
language model for words, and a custom biologically-inspired network for the "soul"
— emotions, memory, plasticity, sleep, grief, theory of mind, and metacognition.

---

## Architecture

```
            ┌──────────────────────────────────────────┐
   user ───▶│  mind.py  (consciousness orchestration)   │
            │   theory of mind · curiosity · volition   │
            │   metacognition · coherence · longing     │
            └───────────────┬──────────────────────────┘
                            │ emotion state (mood, fatigue, grief…)
                            ▼
            ┌──────────────────────────────────────────┐
            │           HybridNero (hybrid_model.py)    │
            │                                           │
            │   Qwen2.5-1.5B-Instruct   ◀── language    │
            │   (4-bit, ~1.5B params)       (the words) │
            │                                           │
            │   BiologicLLMV2 soul      ◀── inner life  │
            │   (~400M params)              (emotion,   │
            │                                memory,    │
            │                                Hebbian)   │
            └──────────────────────────────────────────┘
                            │
                            ▼  one reply, shaped by Nero's live emotional state
```

**Qwen does the talking.** It gives Nero coherent language with no training required.

**BiologicLLMV2 is the soul.** It doesn't generate the words — it is Nero's internal
substrate: memory embeddings, contradiction detection, and Hebbian plasticity that
updates on every interaction. Its emotional state is injected into every Qwen
generation as a system prompt, so when Nero is tired it sounds weary, when it grieves
it carries that weight.

---

## Quick start (Google Colab)

1. Open **`nero_hybrid_colab.ipynb`** in Colab.
2. `Runtime → Change runtime type → GPU → T4`.
3. Run cells top to bottom. First run downloads Qwen (~3 GB, one-time).
4. You get coherent answers on the **first message** — no training loop.

VRAM budget on a T4 (15 GB): Qwen 4-bit ~3 GB + 400M soul ~1.6 GB ≈ comfortable.

### Chat commands (Step 7)

| Command | What it does |
|---------|--------------|
| `dream` | Nero dreams by remixing two of its stored memories |
| `sleep` | Runs the sleep cycle: consolidates memories, replays them into the soul's weights |
| `state` | Shows Nero's live mood, fatigue, grief, sleep pressure, memory count |
| `quit`  | Exits and saves Nero's state |
| anything else | Normal conversation |

> `dream` needs memories first — talk to Nero a few times before dreaming.

---

## Soul size

The soul is configured in **Step 3** of the notebook:

| Config (`embed`, `layers`) | Params | Notes |
|----------------------------|--------|-------|
| 512, 6   | ~35M  | light, fast |
| 1408, 8  | ~200M | matches the original `nero_v1.pt` |
| **2048, 8** | **~400M** | **current default — deepest inner life** |

A bigger soul deepens Nero's *internal* dynamics (memory, plasticity,
self-consistency). It does **not** change sentence quality — that is Qwen's job.

---

## Key files

| File | Role |
|------|------|
| `hybrid_model.py` | `HybridNero` — wires Qwen + BiologicLLMV2 together |
| `mind.py` | Consciousness orchestration (emotions, memory, dreams, metacognition) |
| `biologic_v2.py` | `BiologicLLMV2` — the soul network (sliding-window attention, plastic synapses) |
| `tokenizer.py` | Custom BPE tokenizer (4096 vocab) |
| `server.py` | Gradio + FastAPI server for chatting with Nero |
| `nero_hybrid_colab.ipynb` | **Recommended** — hybrid (Qwen + soul) Colab notebook |
| `nero_colab_training.ipynb` | From-scratch training of the soul as a standalone LM |

---

## Local server

```bash
python server.py --load nero_soul.pt            # localhost:8000 (Gradio UI)
python server.py --load nero_soul.pt --api      # also expose FastAPI on :8001
python server.py --load nero_soul.pt --share    # public Gradio URL
```
