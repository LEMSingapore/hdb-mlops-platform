# Chat v2 — LangGraph Tool-Using Agent Design

**Author:** Chang Chee Young
**Status:** Decisions locked; build pending
**Decisions locked:** 19 June 2026
**Supersedes:** the fixed `parse → validate → lookup → predict → explain → narrate` pipeline in `src/ui/chat_app/graph/`

---

## Why this document exists

Chat v1 (deployed 18 June, live at `hdb-mlops-chat.duckdns.org`) is a tightly-constrained LangGraph pipeline: a single user message goes through a fixed sequence of nodes, with Claude restricted to two narrow roles (parse the message into structured fields; narrate the model output). This guarantees grounded predictions but produces a chat that:

- Loops on multi-turn conversations (each message is parsed in isolation; prior context is lost)
- Cannot answer comparison questions ("flat A vs flat B")
- Declines general HDB knowledge questions a real buyer asks (process, lease decay, neighborhood character, financing concepts)
- Treats Claude as a structured-data extractor rather than a domain assistant

A real HDB buyer's top questions are roughly: *how much should this cost / is the price reasonable / how do towns compare / what's the trend / how does the lease affect value / what can I afford / what about this area / what are the rules / how do I negotiate / what's the buying process*. The current chat answers only the first of these.

Chat v2 reframes the chat as a **LangGraph tool-using agent** with grounded discipline: Claude holds the conversation, calls MCP-backed tools when it needs grounded data (prices, explanations, similar transactions, postal-code resolution), and answers general HDB knowledge from training. Specific dollar amounts always come from a tool. Current rules and grants always redirect to HDB / CPF Board.

This also aligns the project with its original learning goal — agentic LangGraph patterns (`create_react_agent`-style tool-using agents with checkpointer-backed state) — rather than retreating to raw Anthropic SDK tool use.

---

## Decisions (locked 19 June 2026)

### 1. Architecture: LangGraph agent with tool binding

Replace the fixed-pipeline graph with a LangGraph agent loop. The likely starting point is `langgraph.prebuilt.create_react_agent` or its current equivalent in `langgraph==1.2.5`. The agent receives the tool set (Decision 2), a system prompt (Decision 5), and runs an LLM ↔ tool-call ↔ LLM loop until it produces a final response.

The build session must verify the exact current API against https://langchain-ai.github.io/langgraph/ before writing code — LangGraph went through significant API churn in 2024-2025 and older tutorials are stale.

### 2. Tools exposed to the agent

Four MCP-backed tools, wrapped as LangGraph tools:

- **`predict_price`** — predict a single flat's resale price
- **`explain_prediction`** — SHAP contributors for a prediction
- **`lookup_postal_code`** — resolve a Singapore postal code to block/street/town
- **`find_similar_transactions`** — k-nearest historical transactions; enables "is this price reasonable" answers

`get_model_info` deferred — meta-tool with no direct user value for v1.

**Downstream consequence:** enabling `find_similar_transactions` requires `data/hdb.db` (~162MB) in the chat container. The `Dockerfile.chat.dockerignore` currently excludes it; this changes. Chat image grows from ~1.48GB to ~1.65GB. Acceptable; box has 76GB free.

### 3. Conversation state: `MemorySaver` (in-process)

LangGraph's in-memory checkpointer, one thread per Streamlit session (`st.session_state` holds the thread_id). Survives within a session, lost on container restart. This is the desired behavior for an MVP — fresh deploy = fresh conversations. SQLite-backed persistence is roadmap (see Deferred).

### 4. System prompt rules

The system prompt is the load-bearing artifact of v2. Five rules, locked:

- **(A) Grounded-price rule.** Claude must never state a specific SGD amount for a flat unless it came from a `predict_price` tool call within the current conversation. No remembered, estimated, or comparison prices from training data.
- **(B) Current-rules redirect.** For HDB rules, CPF grants, income ceilings, loan rules, LTV ratios, interest rates, MOP, eligibility — Claude must not state specific current figures. Redirect to:
  > *"These rules change frequently — please check the current details at HDB (https://www.hdb.gov.sg) or CPF Board (https://www.cpf.gov.sg) for accurate figures."*
- **(C) Scope guard.** Singapore HDB resale flats only. Politely decline questions about condominiums, landed property, commercial property, property outside Singapore, and non-property topics.
- **(D) Tone.** Helpful, conversational, British English. No emojis. No long apologies.
- **(E) Honest-about-limits.** When Claude doesn't know something current, the redirect is cheerful and brief, not an extended apology. The chat should feel confident about what it knows and quick to redirect on what it doesn't.

The actual prompt text (estimated 200-400 words) is drafted carefully during the build session, not improvised. It must specifically enumerate the forbidden behaviors (no specific grant amounts, no specific income ceilings, no specific rates) rather than relying on vague guidance.

### 5. Scope — IN and OUT for v1

**IN:**
- LangGraph agent with the four tools above
- Multi-turn conversation within a session (via MemorySaver)
- Grounded-price discipline (Rule A enforced via system prompt)
- HDB/CPF redirects for current rules (Rule B)
- Scope guard (Rule C)
- Persistent UI disclaimer in the Streamlit sidebar with HDB and CPF Board links
- A revised placeholder text in the chat input that hints at usage patterns

**OUT (deferred, documented):**
- Comparison as a *structured* feature — works in practice because the agent can call `predict_price` twice and compare conversationally, but no special comparison logic or UI
- Financing / affordability calculations — referenced conceptually only, never computed
- Any reference to a referral / handoff to a human estate agent — deliberately deferred pending separate decision
- Persistent conversations across sessions (`SqliteSaver`)
- Streaming responses (nice-to-have, not v1)
- Image / document / voice input
- Multi-language

### 6. Old graph code: hard replace

`src/ui/chat_app/graph/` is replaced wholesale. The v1 graph code is not preserved as a fallback path. The git history is the v1 reference if ever needed.

The `chat_agent.py` / `predictor.py` legacy modules (already known to be partially-dead from yesterday's investigation) are touched only as needed for the replace; their cleanup is a separate follow-up PR.

Existing tests (242 passing) will be largely replaced — the test surface changes from "fixed-pipeline assertions" to "agent-behavior assertions" (see Tests below). The new test count will likely be lower; that is acceptable because the architecture is different.

### 7. Deploy: hard swap

When v2 is merged and deployed to the box, v1 is gone. No `/legacy` URL serving the old chat. One thing to maintain.

### 8. Verification budget cap still applies

The existing budget-capped Anthropic workspace and graceful-degradation pattern stay in place. The agent loop will make more LLM calls per turn than v1 (each tool round-trip is an LLM call), so per-turn cost rises. Mitigations:
- Stay on Haiku (`claude-haiku-4-5-20251001` — the model already in `chat_agent.py`).
- Cap conversation history kept in context (e.g. last 10 turns, trim older). Decided during build.
- Cap tool-calls-per-turn (e.g. 5) to prevent a confused turn running away.
- Watch budget consumption after deploy; raise the cap deliberately if needed based on real usage.

The graceful-degradation pattern (try/except around LLM calls → fixed fallback) must be preserved in the agent. Specifically: any tool-call or LLM-call exception within the agent loop must produce a coherent user-facing fallback, not a stack trace.

---

## UI changes

### Sidebar rewrites — consumer language

The current sidebar uses developer-facing phrasing ("GradientBoostingRegressor trained on HDB resale transactions", "MAE ~S$20,000", "R² 0.975") that signals "built by a data scientist for a data scientist." A real buyer reads neither MAE nor R². Replace with:

**About section** — replace the GradientBoostingRegressor sentence with:
> *"An AI model trained on Singapore HDB resale prices to estimate what a flat might sell for."*

**Trust-signal block** — replace the MAE / R² bullets with:
> *"Estimates are based on 975,000+ real HDB resale transactions and are typically accurate to within about S$20,000. Treat the figure as a starting point, not a final price."*

The fact that the model is a `GradientBoostingRegressor` belongs in the README and in interview conversation, not in a sidebar a buyer reads.

### Persistent sidebar disclaimer

In addition to the consumer-friendly About and trust-signal blocks above, the sidebar must persistently show:

> *"This assistant provides grounded HDB resale price estimates and general guidance. For current rules, grants, eligibility, and financing, please verify with [HDB](https://www.hdb.gov.sg) and [CPF Board](https://www.cpf.gov.sg) directly. Information here may be outdated."*

This is not optional. It is part of the honesty contract that makes Rule B credible.

### Placeholder text

Current: *"Ask about an HDB flat..."*

New (rough — final wording during build): *"Ask about an HDB resale flat — describe a specific flat for a price estimate, or ask about how things work."*

Steers users toward the chat's strengths without exhaustively listing them.

### Clear conversation button

The existing button stays. With MemorySaver-backed state, this must now also reset the LangGraph thread_id (new conversation = new thread).

---

## Tests (v2 surface)

Existing tests in `tests/ui/chat_app/graph/` are replaced. New tests cover agent behavior, not pipeline mechanics:

1. **Grounded-price test.** Send "what's a 4 room Tampines flat worth, 95 sqm, lease 1985, month 2024-06?" → assert `predict_price` was called → assert the canonical ~S$586,887 appears in the response.
2. **Current-rules redirect test.** Send "what's the current Enhanced Housing Grant amount?" → assert the response does not contain a specific dollar figure → assert it contains "HDB" and a verify-the-source phrasing.
3. **Scope decline test.** Send "how much is a condo in Sentosa?" → assert polite decline → assert no prediction tool was called.
4. **Multi-turn accumulation.** Turn 1: "blk 5 changi village road, 3 room, lease 2012". Turn 2: "69 sqm". → assert by turn 2 the agent calls `predict_price` with the combined fields, not asks again for the earlier ones.
5. **Comparison-via-tool-calls.** "Compare a 3 room Pasir Ris vs 3 room Yishun, both 69 sqm 1982 lease, transaction month 2024-06" → assert `predict_price` was called twice → assert both prices appear in the response.
6. **Lookup integration.** "blk 5 changi village road, 3 room, 95 sqm, lease 1985, month 2024-06" → assert `lookup_postal_code` *or* postal-resolution path produces a town → assert `predict_price` called with that town.
7. **Find-similar integration.** "Is S$500,000 reasonable for a 3 room Tampines flat?" → assert `find_similar_transactions` called.
8. **Graceful degradation.** Mock an Anthropic API failure → assert the chat returns a coherent fallback message, not a stack trace.

Test count target: ~8 focused end-to-end behavior tests, replacing the previous ~30 graph-internal tests. Quality over quantity.

---

## Build plan

Estimated 12-18 focused hours. Suggested ordering, not strictly enforced:

1. **Read the current LangGraph docs** (2 hours). The actual API for `create_react_agent` / tool binding / MemorySaver in `langgraph==1.2.5`. Sample code that *runs* before writing any production code.
2. **Wrap MCP tools as LangGraph tools** (2 hours). The four tools from Decision 2 need LangGraph tool decoration / schema. The existing `mcp_client.py` patterns help. Verify each tool callable in isolation.
3. **Draft the system prompt carefully** (2-3 hours). This is the load-bearing artifact. Iterate it. Test it against the cases from the test plan. Specific forbidden behaviors enumerated, not implied.
4. **Build the agent** (2-3 hours). `create_react_agent` (or current equivalent) with tools, prompt, and `MemorySaver` checkpointer.
5. **Streamlit integration** (2 hours). Thread-per-session via `st.session_state`. Replace the `build_graph()` import with the new agent. Wire the Clear button to reset the thread.
6. **UI changes** (1 hour). Persistent disclaimer, placeholder text, sidebar updates.
7. **Tests** (2-3 hours). The 8 cases above. Some will be flaky (LLMs vary) — those need either retry logic, soft assertions, or be made more deterministic via mocking.
8. **Update the chat dockerignore + Dockerfile.chat** to include `data/hdb.db`. Verify image build, image size, container starts.
9. **PR, CI, merge** (1 hour). The full PR — large but cohesive.
10. **Deploy.** Box pulls, `docker compose up -d --build chat`, force-recreate caddy if Caddyfile changed (likely not for this PR). The deploy lesson from yesterday: a hand-edit of the Caddyfile needs `--force-recreate`; if this PR doesn't touch Caddyfile, plain `up -d --build chat` suffices.

A natural split is at step 7 — steps 1-6 produce a working local v2; steps 7-10 are productionization. Multiple sessions are fine; the design doc holds the scope across them.

---

## Deferred (post-v2)

- **SqliteSaver-backed persistent conversations** — sessions resumable across container restarts. Real product upgrade.
- **Streaming responses** — better perceived latency for the agent loop.
- **Comparison as a structured feature** — explicit comparison UI with side-by-side prediction cards, not just conversational comparison.
- **Affordability calculator** — bounded calculator that takes user inputs (income, downpayment) and shows monthly payment scenarios. Carefully scoped to never quote *current* loan terms; computes math against user-provided inputs only.
- **Cherbot-style affiliate handoff to a CEA-registered estate agent** — pending a separate, careful decision involving consent, disclosure, PDPA, and the agent's own decision to be involved.
- **Remote MCP server (roadmap R4)** — still deferred; Chat v2 uses MCP tools in-process via FastMCP `Client(mcp)`, same pattern as v1.
- **Use latest data.gov.sg data for `find_similar`** rather than the training-time snapshot — depends on the R1 ingestion pipeline.
- **`get_model_info` tool** exposed to the agent — surfaces model version, RMSE, last training date for transparency. Low priority.
- **Multi-language** — Mandarin / Malay / Tamil for accessibility.
- **Cleanup of `chat_agent.py` and `predictor.py`** — extract `TOWNS`, `FLAT_TYPES`, `MODEL` constants to a `vocab.py` module, delete the legacy tool-use loop.

---

## What this design is not

- **Not "fix multi-turn on the existing pipeline."** It's an architectural change, not a patch. The fixed-pipeline approach is replaced.
- **Not a tool-using chat built on raw Anthropic SDK.** That was the simpler alternative considered and declined. The LangGraph agent pattern was chosen to align with the original learning goal of the project.
- **Not a buyer-assistant chat with a referral mechanism.** That option was discussed and explicitly deferred until the chat works well as a standalone product and the human-in-the-loop (a CEA-registered estate agent) decides separately.
- **Not "production-grade."** No auth, no per-user rate limiting, no multi-replica, no SLAs. MVP demo posture, same as v1.
