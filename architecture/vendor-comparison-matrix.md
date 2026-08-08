# AI Vendor Comparison Matrix

A practical reference for choosing vendors at each layer of an AI system.
Built as part of learning AI solutions architecture.

Last updated: July 2026

---

## Layer 1 — LLM Providers (Closed API)

Use when: you want to call an external API and not manage models yourself.

| Provider | Key Models | Best For | Pricing | Strengths | Watch Out For |
|----------|-----------|---------|---------|-----------|---------------|
| Anthropic | Claude Opus 4, Sonnet 4, Haiku | Complex reasoning, long context, enterprise safety | Per token (input/output) | Best at instruction following, lowest hallucination rate, 200K context | Higher cost than OpenAI at scale |
| OpenAI | GPT-4o, GPT-4.1, o3 | General purpose, coding, broad ecosystem | Per token | Largest ecosystem, best tooling, function calling | Rate limits, cost at high volume |
| Google | Gemini 2.0 Flash, Pro | Multimodal, Google Workspace integration | Per token | Fast, cheap Flash tier, native multimodal | Less mature enterprise support |
| Groq | Llama 3, Mixtral via Groq API | Speed-critical applications | Per token | Fastest inference available (LPU hardware), very low latency | Limited model selection, not for fine-tuning |
| Mistral | Mistral Large, Small | European data residency, open weights option | Per token | GDPR-friendly, strong European enterprise option | Smaller ecosystem |

---

## Layer 1 — LLM Providers (Open Weight / Self-Hosted)

Use when: data privacy requirements, cost at scale, or need to fine-tune.

| Model Family | Key Models | Best For | Hardware Needed | Strengths | Watch Out For |
|-------------|-----------|---------|-----------------|-----------|---------------|
| Meta Llama | Llama 3.1 8B, 70B, 405B | General purpose self-hosting | 8B: 1× GPU, 70B: 4× GPU | Best open-weight quality, large community | Large 70B+ needs serious hardware |
| Mistral | Mistral 7B, Mixtral 8×7B | Efficient self-hosting, European compliance | 1-2× GPU | MoE architecture is efficient, good quality | Less support than Llama |
| Microsoft Phi | Phi-3 Mini, Medium | Edge/mobile deployment, low VRAM | CPU or 1 small GPU | Tiny but surprisingly capable, fits consumer hardware | Not for complex reasoning tasks |
| Google Gemma | Gemma 2B, 7B | Research, lightweight deployment | 1× GPU or CPU | Lightweight, Apache licensed | Limited instruction following vs Llama |

---

## Layer 2 — Vector Databases

Use when: building RAG pipelines, semantic search, or similarity retrieval.

| Database | Best For | Hosting | Free Tier | Scales To | Strengths | Watch Out For |
|----------|---------|---------|-----------|-----------|-----------|---------------|
| Qdrant | Production RAG, hybrid search | Cloud + self-host | Yes (local) | Billions of vectors | Rust-based (fast), hybrid search (dense + sparse), best-in-class filtering | Newer, smaller community than Pinecone |
| Pinecone | Enterprise RAG, managed service | Cloud only | Yes (limited) | Billions of vectors | Easiest to get started, fully managed, battle-tested | Vendor lock-in, expensive at scale |
| ChromaDB | Local development, prototyping | Local + cloud | Yes | Millions of vectors | Simplest setup, great for dev/test | Not production-ready at scale |
| pgvector | Adding vector search to existing Postgres | Self-host | Yes | Tens of millions | No new infra if already using Postgres, SQL joins with vectors | Slower than purpose-built at large scale |
| Weaviate | Multi-modal search, GraphQL API | Cloud + self-host | Yes | Billions of vectors | Built-in ML models, multi-modal support | Complex setup, steeper learning curve |
| Milvus | Large-scale enterprise | Self-host (Zilliz cloud) | Yes | Billions of vectors | Battle-tested at massive scale (used by Salesforce, eBay) | Operationally complex |

---

## Layer 3 — Inference Serving Frameworks

Use when: self-hosting open-weight models in production.

| Framework | Best For | GPU Required | Key Feature | Throughput | Ease of Use | I've Used It |
|-----------|---------|-------------|-------------|-----------|------------|----------------|
| Ollama | Local development, demos | No (CPU works) | Dead-simple model management | Low (sequential) | ★★★★★ | ✓ Benchmarked |
| vLLM | Production serving, high concurrency | Yes | PagedAttention + continuous batching | Very high | ★★★☆☆ | ✓ Benchmarked (99× vs Ollama) |
| TensorRT-LLM | Maximum NVIDIA GPU performance | Yes (NVIDIA only) | Kernel fusion, INT8/FP8 quantization | Highest | ★★☆☆☆ | Not yet |
| llama.cpp | CPU inference, edge deployment, GGUF | No | GGUF quantization, runs on laptops | Medium | ★★★★☆ | Read README only |
| HuggingFace TGI | HuggingFace ecosystem, quick deploy | Yes | Tensor parallelism, streaming | High | ★★★★☆ | Not yet |
| SGLang | Complex multi-turn, agentic workflows | Yes | RadixAttention for prefix sharing | High | ★★★☆☆ | Not yet |

---

## Layer 4 — Orchestration / Agent Frameworks

Use when: building multi-step AI workflows, RAG pipelines, or agents.

| Framework | Best For | Language | Key Feature | I've Used It |
|-----------|---------|---------|-------------|----------------|
| LangChain | RAG pipelines, chains, agents | Python/JS | Largest ecosystem, most integrations | Indirectly |
| LangGraph | Stateful multi-agent workflows | Python | Graph-based agent orchestration, cycles | ✓ Production (Conflux) |
| LlamaIndex | Data connectors, RAG-focused | Python | Best for ingestion pipelines, document QA | Know it |
| CrewAI | Multi-agent collaboration | Python | Role-based agents, task delegation | Heard of it |
| Semantic Kernel | Microsoft/Azure AI integration | Python/C# | Enterprise-grade, Azure native | Know it |

---

## Quick Decision Guide

**Choosing an LLM provider:**
- Need best quality + long context → Anthropic Claude
- Need broadest ecosystem + tools → OpenAI
- Need lowest latency → Groq
- Need data privacy / self-host → Llama 3 via vLLM
- Operating in EU with compliance needs → Mistral

**Choosing a vector DB:**
- Just prototyping → ChromaDB (local)
- Already on Postgres → pgvector
- Production RAG, want managed → Pinecone
- Production RAG, want control + hybrid search → Qdrant
- Massive scale (billions of vectors) → Milvus

**Choosing an inference framework:**
- Local dev / demo → Ollama
- Production, high concurrency, GPU → vLLM
- Maximum GPU performance, NVIDIA infra → TensorRT-LLM
- No GPU, edge/laptop → llama.cpp

---

*Part of AI solutions architecture learning: [anirudhs16.github.io/ai-systems](https://anirudhs16.github.io/ai-systems)*
