# Enterprise AI Architecture Case Study
## Customer Support AI System

**Scenario:** Mid-size SaaS company, 500 employees, 10,000 customers  
**Scale:** 10,000 queries per day  
**Date:** August 2026

---

## What the system does

A customer support AI system that handles queries across chat and voice channels. From the customer's perspective: ask a question, get an accurate answer, take an action (refund, cancellation, appointment change) — or get handed to a human agent when the situation requires it.

The system operates within defined business rules. It knows when it can autonomously process a refund vs when it needs human approval. It knows when to escalate. These rules are not optional guardrails — they are the core of what makes the system production-safe.

---

## Architecture overview

```
Customer (chat / voice)
        ↓
   Orchestration layer (LangGraph)
        ↓
  ┌─────────────────────────────┐
  │   LLM (Claude via API)      │
  │   + Tool calling            │
  └─────────────────────────────┘
        ↓                ↓
  Knowledge retrieval   Live system APIs
  (Qdrant vector DB)    (Orders, CRM, Billing)
        ↓
  Human escalation
  (when needed)
```

---

## Components and vendor choices

### LLM — Anthropic Claude (Sonnet)

**Choice:** Claude Sonnet via Anthropic API  
**Why:** Claude's instruction following is the strongest available for policy-constrained use cases. When the system prompt says "do not process refunds over $500 without human approval," Claude respects that constraint reliably. Hallucination rate on out-of-scope queries is lower than alternatives — critical when wrong answers create legal liability. The 200K context window handles long customer conversation histories without truncation.

**Why not self-hosted Llama:** Data privacy is manageable via Anthropic's enterprise data processing agreement. The operational cost of self-hosting and maintaining a 70B model exceeds the API cost at 10K queries/day at this company's scale.

**Cost estimate:** At 10K queries/day, averaging 1,500 tokens per query (input + output), Claude Sonnet at ~$3/million tokens = approximately **$45/day or ~$1,350/month** in LLM costs alone.

---

### Vector Database — Qdrant

**Choice:** Qdrant Cloud  
**Why:** Customer support queries benefit from hybrid search — combining semantic similarity (embedding-based) with keyword matching. A query like "return policy for orders placed during the sale" needs both semantic understanding and keyword precision. Qdrant's hybrid search (dense + sparse vectors combined) outperforms embedding-only search for this pattern.

**Knowledge sources ingested:**
- Company policy documents (refund policy, cancellation terms, SLAs)
- Product documentation and FAQs
- Past resolved support tickets (anonymized)
- Agent playbooks and escalation guidelines

**Retrieval flow:** Customer query → embed query → Qdrant hybrid search → top 5 relevant chunks → passed to Claude as context.

**Scale:** At 10K queries/day, retrieval adds ~50-100ms latency. Qdrant Cloud free tier handles up to 1GB of vectors — sufficient for most mid-size company knowledge bases. Paid tier at ~$25/month for production reliability.

---

### Orchestration — LangGraph

**Choice:** LangGraph  
**Why:** Customer support is a stateful, multi-step workflow — not a single prompt-response. A customer conversation might span: greeting → query classification → knowledge retrieval → tool call to orders API → response generation → escalation check → human handoff. LangGraph's graph-based state machine makes this flow explicit, debuggable, and modifiable without rewriting everything.

**Key graph nodes:**
- `classify_intent` — determine query type (information, action request, complaint)
- `retrieve_knowledge` — Qdrant hybrid search
- `call_tool` — live API calls (orders, appointments, billing)
- `check_escalation` — rule-based escalation logic
- `generate_response` — Claude with retrieved context
- `human_handoff` — pass conversation state to human agent queue

**Why not LangChain alone:** LangChain's sequential chains don't handle the branching and state persistence that a real support conversation requires. LangGraph's explicit state graph is the right tool for workflows that cycle, branch, and need to be interrupted for human review.

---

### Live system integrations (tool calling)

The LLM calls these as tools when queries require real-time data:

| Tool | What it does | Latency target |
|------|-------------|----------------|
| `get_order_status` | Pulls live order data from OMS | <200ms |
| `get_appointment` | Checks appointment availability | <200ms |
| `process_refund` | Initiates refund (within rules) | <500ms |
| `cancel_order` | Cancels order (within rules) | <500ms |
| `escalate_to_human` | Routes to human agent queue | <1s |

All tool calls are logged for audit. Refund and cancellation tools have hard limits enforced at the API layer — not just in the prompt.

---

## Cost estimate — 10,000 queries/day

| Component | Cost/month |
|-----------|-----------|
| Claude Sonnet API | ~$1,350 |
| Qdrant Cloud | ~$25 |
| Infrastructure (hosting, logging) | ~$200 |
| Human escalation (est. 15% of queries) | Variable (existing agent cost) |
| **Total AI infrastructure** | **~$1,575/month** |

At 10K queries/day, that is **$0.005 per query** in AI infrastructure costs — before human escalation. If human agents handle 15% of queries (1,500/day), the cost of those interactions dwarfs the AI infrastructure cost and represents the primary financial incentive to improve containment rate.

---

## Top 3 failure modes

### 1. Inaccurate responses causing policy violations

**What happens:** The system confidently answers a refund or cancellation query based on retrieved context that is outdated, mismatched to the customer's specific situation, or hallucinated. Customer acts on the wrong information. Legal and reputational risk.

**Mitigation:**
- Retrieval confidence threshold — if top retrieved chunk similarity score is below 0.75, respond with "I need to check on that" and escalate rather than generate a low-confidence answer
- All refund/cancellation actions confirmed via tool call to live system, not generated as text
- Monthly human review of 100 random conversations to catch systematic errors
- Policy documents versioned — when policy changes, old chunks removed and re-ingested immediately

---

### 2. Escalation rate miscalibration

**What happens:** System escalates too aggressively — human agents are overwhelmed and the AI provides no cost benefit. Or system escalates too rarely — customers are stuck in loops with an AI that cannot solve their problem, leading to churn and negative reviews.

**Mitigation:**
- Weekly monitoring of escalation rate (target: 10-20% of queries)
- Sentiment detection before escalation check — angry or frustrated signals trigger earlier escalation regardless of query type
- Post-conversation CSAT (customer satisfaction) score tracked by whether query was AI-resolved or human-resolved
- Escalation threshold tuned monthly based on CSAT delta between the two paths

---

### 3. No ownership in production

**What happens:** The system launches. Nobody owns it operationally. Model behavior drifts as company policies change but the knowledge base is not updated. Retrieval quality degrades. Nobody notices until customers start complaining at scale.

**Mitigation:**
- Named owner for the AI system — not a committee, one person responsible
- Weekly knowledge base audit: were any policies updated this week? If yes, re-ingest
- Production monitoring dashboard: query volume, escalation rate, CSAT, average retrieval score — reviewed weekly
- On-call runbook: what to do if escalation rate spikes above 40% (kill switch to route all queries to human agents while investigating)

---

## What I would validate before going to production

1. **Retrieval quality:** Run 50 real historical support queries through the system. What percentage get the right chunks retrieved? Target: >85%.

2. **Policy compliance:** Create 20 adversarial test cases — customers trying to get refunds outside policy, asking for exceptions. Does the system hold the line correctly every time?

3. **Latency:** End-to-end response time including retrieval + tool calls + LLM generation. Target: <3 seconds for 95th percentile. Customers abandon if response takes longer.

4. **Escalation accuracy:** Of queries that went to human agents historically, would the system have escalated them? Of queries agents resolved quickly, would the system have contained them?

---

*Part of AI solutions architecture portfolio: [anirudhs16.github.io/ai-systems](https://anirudhs16.github.io/ai-systems)*  
*Vendor reference: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)*
