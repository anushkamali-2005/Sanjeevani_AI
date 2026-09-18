# Darukaa.Earth Biodiversity Intelligence Chatbot — Full Build Blueprint

**Purpose of this document:** a start-to-finish playbook for building this project so it scores at the top of all five evaluation criteria. It covers architecture, knowledge base design, conversation design, recommendation logic, UI/UX direction (based on real official biodiversity/environment portals), a day-by-day build plan, and exact prompts you can paste into an AI coding assistant (Claude Code, Cursor, etc.) at each step. CI/CD and deployment are intentionally left out, as requested — everything up to "runs correctly on your machine and is demo-able" is covered.

---

## 0. How the evaluators will actually grade you — reverse-engineering the rubric

| Criterion | Weight | What "best in the pool of 800" looks like |
|---|---|---|
| Depth of Reasoning | 30% | Recommendations that chain 2–3 variables together and state a mechanism, not a fact. Never "plant trees" — always "X causes Y via mechanism Z, which raises metric M by N% over T time." |
| Scientific Grounding | 25% | Every claim traceable to a named source (FAO, IPCC, IPBES, a specific paper) with a real number range, not an invented one. |
| Knowledge System Design | 20% | A visible, inspectable retrieval pipeline (embeddings + vector DB), not "we prompted GPT with a big system prompt." |
| Conversational Intelligence | 15% | Genuine multi-turn state: the bot remembers earlier answers, only asks for what's still missing, and updates its recommendation when new data arrives. |
| Output Clarity | 10% | Every answer follows one strict structured template (recommendation / mechanism / metrics impacted / time horizon / confidence / source) — never freeform prose. |

**The single biggest score risk** for most of the 800 entrants: they will wire an LLM to a system prompt and call it "RAG." Your differentiator is a *real, inspectable* knowledge layer (Section 2) plus a *reasoning graph* that explicitly links variables (Section 4) — both things a judge can open and verify in under two minutes.

---

## 1. System Architecture

```
                        ┌─────────────────────────────┐
                        │        Frontend (UI)         │
                        │  Chat + structured input form│
                        └───────────────┬───────────────┘
                                        │ REST / WebSocket
                        ┌───────────────▼───────────────┐
                        │       Conversation Manager     │
                        │  - session/thread memory       │
                        │  - slot-filling (missing vars) │
                        │  - intent + entity extraction  │
                        └───────────────┬───────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                          │
   ┌──────────▼──────────┐   ┌──────────▼──────────┐   ┌──────────▼──────────┐
   │   Retrieval Layer    │   │  Reasoning Engine    │   │   Input Validators   │
   │ (RAG over vector DB) │   │ (multi-metric graph) │   │ (text / JSON / geo)  │
   └──────────┬──────────┘   └──────────┬──────────┘   └──────────┬──────────┘
              │                         │                          │
              └─────────────┬───────────┴──────────────────────────┘
                            │
                 ┌──────────▼──────────┐
                 │   LLM Orchestrator   │
                 │ (prompt assembly +   │
                 │  structured output)  │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  Structured Response  │
                 │  Renderer (UI cards)  │
                 └──────────────────────┘
```

**Core components you must build:**
1. Knowledge base ingestion pipeline (documents → chunks → embeddings → vector DB)
2. Retriever (semantic search + metadata filters by variable type: soil / climate / land-use / biodiversity / human-impact)
3. Variable-relationship graph (a small structured knowledge graph, not just vector search — this is what makes reasoning "multi-metric" instead of single-lookup)
4. Conversation state machine (tracks which of the 5 required variable categories are known/unknown per session)
5. Recommendation generator (LLM call constrained to a strict output schema, grounded in retrieved chunks + graph edges)
6. Frontend chat + structured-input UI

---

## 2. Knowledge System Design (20% of grade — make this the most "inspectable" part of your repo)

### 2.1 What "knowledge system," not "prompt," means to a judge
A judge should be able to:
- Open a `/knowledge` folder and see real source documents or structured datasets.
- Open a `retrieval.py` (or equivalent) and see embeddings being generated and queried against a vector index.
- See, in every chatbot answer, which chunk(s)/source(s) were retrieved and used (a "Sources used" trace in the response).

### 2.2 Recommended tech choices
- **Vector DB:** ChromaDB (local, zero-infra, embeds directly in Python) or Qdrant (if you want a slightly more "production" look for judges). Chroma is the fastest to get working correctly, which matters more than impressiveness of the DB name.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (free, local, no API cost, fast) or OpenAI/Anthropic embeddings if you have API budget. For a hackathon judged on reasoning, local embeddings are perfectly sufficient — don't burn time/money here.
- **LLM for reasoning/generation:** Claude (via API) or GPT-4-class model, used only for the final synthesis step — never for "looking up facts," which must come from retrieval.
- **Backend:** Python + FastAPI (clean, fast, easy to demo with `/docs` auto-generated Swagger UI — bonus point for "structured input (JSON)" requirement since FastAPI gives you this for free).
- **Structured knowledge graph:** a simple JSON/YAML file or a NetworkX graph in Python — do not over-engineer this into Neo4j unless you already know it well.

### 2.3 What to actually put in the knowledge base
Build **two layers**, not one:

**Layer A — Unstructured evidence corpus (for RAG / vector search).**
Ingest real, citable content chunks (a paragraph or two each, with a source tag) covering:
- Soil organic carbon & cover cropping / agroforestry effects (FAO "Soils are the foundation..." reports, FAO 4 per 1000 initiative)
- IPCC AR6 WG2 chapters on land degradation, biodiversity-climate interactions
- IPBES Global Assessment (2019) — drivers of biodiversity loss, land-use change, pollinator decline
- FAO "The State of the World's Biodiversity for Food and Agriculture"
- Studies on hedgerows/field margins and pollinator/species richness (many open-access papers exist — search Google Scholar / FAO / IPBES public PDFs)
- Studies on riparian buffer strips and water quality / species survival
- Studies on monoculture vs. polyculture/intercropping and habitat fragmentation

For each chunk, store metadata:
```json
{
  "id": "doc_014_chunk_3",
  "text": "...",
  "source": "FAO (2017), Soil Organic Carbon: the hidden potential",
  "url_or_citation": "FAO, 2017",
  "topic_tags": ["soil", "carbon", "cover_crops"],
  "linked_metrics": ["soil_organic_carbon", "microbial_diversity"]
}
```

**Layer B — Structured relationship graph (this is your secret weapon for "multi-metric reasoning").**
A small, explicit graph of variable → intervention → effect → evidence, e.g.:

```json
{
  "interventions": [
    {
      "name": "Legume-based cover cropping",
      "affects_metrics": ["soil_organic_carbon", "microbial_diversity", "pollinator_support"],
      "mechanism": "Nitrogen fixation + continuous root biomass increases organic matter input and feeds soil microbial communities, which in turn support higher trophic diversity.",
      "effect_size": "+15-25% soil organic carbon over 2-3 years",
      "preconditions": ["low_to_moderate_soil_organic_carbon", "any_rainfall_regime"],
      "time_horizon": "medium_term",
      "source": "FAO"
    },
    {
      "name": "Agroforestry / intercropping in semi-arid monoculture systems",
      "affects_metrics": ["soil_organic_carbon", "habitat_diversity", "species_richness", "water_retention"],
      "mechanism": "Tree/shrub layers reduce wind erosion and evapotranspiration, add leaf litter carbon, and create vertical habitat structure absent in monocultures.",
      "effect_size": "measurable increases in canopy-dependent species richness within 3-5 years",
      "preconditions": ["monoculture", "semi_arid", "low_rainfall"],
      "time_horizon": "medium_to_long_term",
      "source": "FAO, IPCC AR6 WG2"
    }
  ]
}
```

This graph is what lets your reasoning engine say "because your soil carbon is low AND rainfall is low AND land use is monoculture, these three combine to point at agroforestry, not just one variable in isolation" — that is literally the "Multi-Metric Reasoning" grading criterion, made explicit and inspectable in your code instead of hidden inside an LLM's implicit reasoning.

### 2.4 Retrieval pipeline — exact steps
1. Chunk documents (300–500 tokens per chunk, with overlap of ~50 tokens).
2. Embed each chunk with the sentence-transformer model.
3. Store in Chroma with metadata (topic tags, linked metrics, source).
4. At query time: embed the user's combined context (their stated variables + question) → retrieve top-k (k=5–8) chunks filtered by relevant topic tags.
5. Separately query the structured relationship graph for interventions whose `preconditions` match the user's stated variable values.
6. Pass both (retrieved text chunks + matched graph interventions) into the LLM prompt as grounding context — the LLM's job is only to *synthesize and phrase*, never to invent facts.

---

## 3. Conversational Intelligence (15%)

### 3.1 Required variable slots
Define a fixed schema of "slots" the system needs before it can give a full recommendation:
```
soil_organic_carbon_pct
soil_ph
soil_moisture
land_use_type
rainfall_pattern
region_climate_zone
species_richness_observation (optional/qualitative)
pollution_or_deforestation_pressure (optional)
geo_coordinates (optional, bonus)
```

### 3.2 State machine logic
- On each user turn, run entity extraction (regex + LLM function-calling / structured extraction) to fill any slots mentioned.
- After updating slots, check which **mandatory** slots (soil carbon, land use, rainfall, region) are still empty.
- If any mandatory slot is empty → ask a clarifying question for the single most decision-relevant missing slot (don't ask for everything at once — ask what changes the recommendation the most first, e.g., land use type before soil pH).
- Once enough slots are filled (define a minimum: at least 3 environmental variables, per the challenge's explicit constraint), proceed to the reasoning + retrieval stage.
- If the user updates a variable mid-conversation (e.g., "actually rainfall is moderate, not low"), the state machine must overwrite that slot and note in the reply that the recommendation has been revised.

### 3.3 Memory implementation
- Keep a per-session object: `{slots: {...}, conversation_history: [...], last_recommendation: {...}}`.
- Persist per-session state in memory (a Python dict keyed by session ID) for the hackathon demo — no database needed unless you want to show persistence across page reloads, in which case SQLite is enough.

---

## 4. Evidence-Backed Recommendations & Multi-Metric Reasoning (30% + 25% combined — this is where you win or lose)

### 4.1 The mandatory output schema (never deviate from this)
Every single response must render as:
```json
{
  "recommendation": "Introduce legume-based cover crops in rotation with wheat",
  "mechanism": "Legumes fix atmospheric nitrogen and maintain continuous root biomass, increasing organic matter turnover and feeding soil microbial communities.",
  "metrics_impacted": [
    {"metric": "soil_organic_carbon", "direction": "increase", "magnitude": "15-25% over 2-3 years"},
    {"metric": "microbial_diversity", "direction": "increase", "magnitude": "qualitative increase"},
    {"metric": "pollinator_support", "direction": "increase", "magnitude": "qualitative increase"}
  ],
  "time_horizon": "medium_term (2-3 years)",
  "confidence": "high",
  "source": "FAO Soil Organic Carbon reports; FAO 4 per 1000 initiative",
  "connected_variables": ["soil_organic_carbon", "land_use_type", "rainfall_pattern"]
}
```
Render this as a clean card in the UI (not a paragraph) — this directly satisfies the "Output Quality" rubric line by line.

### 4.2 Reasoning algorithm (how the backend actually produces this, step by step)
1. Take filled slots (e.g., soil_organic_carbon=0.3%, rainfall=low, land_use=monoculture_wheat, region=semi_arid).
2. Query the relationship graph (Section 2.3) for interventions whose `preconditions` overlap with ≥2 of the user's variable values — this enforces genuine multi-variable reasoning instead of single-lookup.
3. Rank candidate interventions by number of overlapping preconditions + number of `affects_metrics` that match variables the user cares about.
4. Retrieve supporting text chunks from the vector DB for the top 1–2 ranked interventions.
5. Construct the LLM prompt (see Section 6.2 for the exact prompt template) with: user's variables, the selected intervention's structured fields, and the retrieved text chunks as grounding.
6. LLM outputs strictly the JSON schema in 4.1 — validate it against the schema in code (e.g., with `pydantic`) before showing it to the user; if validation fails, retry once with a stricter reformatting instruction.

### 4.3 Non-obvious recommendation checklist (use this to self-grade before submitting)
- [ ] Does it name a specific practice, not a category ("legume cover crops," not "sustainable farming")?
- [ ] Does it state a mechanism (the *why*), not just the *what*?
- [ ] Does it cite a real source with a real number range?
- [ ] Does it connect at least 2 of the 5 required variable categories?
- [ ] Would a domain expert nod, or would they say "that's textbook-obvious"? If obvious, add a secondary, less-common intervention as an alternative (e.g., not just "plant trees" but "stagger tree-planting with contour bunding to also address water retention in the same semi-arid, low-rainfall context").

---

## 5. Input Handling

- **Text input:** free-form chat, parsed via LLM-based entity/slot extraction.
- **Structured input (JSON):** expose a `/analyze` POST endpoint that accepts:
```json
{
  "soil_organic_carbon_pct": 0.3,
  "rainfall_pattern": "low",
  "land_use_type": "monoculture_wheat",
  "region_climate_zone": "semi_arid",
  "geo_coordinates": {"lat": 26.9, "lon": 75.8}
}
```
This endpoint should run the exact same reasoning pipeline as the chat, just skipping the slot-filling conversation. This one endpoint satisfies "Structured input" and "Input Handling" cleanly and is easy to demo via Swagger UI (`/docs`) — visually impressive to a judge in 10 seconds.
- **Geo-coordinates (bonus):** if provided, do a simple lookup (even a static table or a free API like Open-Meteo for climate, or a static Köppen climate classification lookup) to auto-fill `region_climate_zone` and `rainfall_pattern` if the user hasn't stated them — this is an easy, high-visibility bonus feature.

---

## 6. Exact Prompts To Use (paste these into your AI coding assistant in this order)

> Use these with Claude Code, Cursor, or directly in a Claude/GPT chat as you build. They are written to be copy-pasted verbatim, one at a time, in the given sequence. After each step, review the output before moving to the next prompt.

### 6.1 Project scaffolding prompt
```
Set up a Python project for an AI biodiversity intelligence chatbot with this structure:

/backend
  /app
    main.py              # FastAPI app entrypoint
    conversation.py       # conversation state machine + slot filling
    retrieval.py           # embeddings + Chroma vector DB logic
    reasoning.py            # relationship graph + intervention ranking
    schemas.py               # pydantic models for structured input/output
    llm_client.py              # wrapper around the LLM API call
  /knowledge
    /documents               # raw text/pdf source chunks
    graph.json                 # the structured intervention relationship graph
  requirements.txt
  README.md
/frontend
  (a clean single-page chat UI, plain HTML/CSS/JS or React — I'll specify design separately)

Use FastAPI, ChromaDB, sentence-transformers, and pydantic. Do not set up any deployment, Docker, or CI/CD — just a working local dev setup with clear run instructions in the README (uvicorn for backend, and how to open the frontend).
```

### 6.2 Knowledge base ingestion prompt
```
Write a Python script `ingest.py` that:
1. Reads text chunks from /knowledge/documents (I will provide ~20-30 chunks as .txt or .json files, each with fields: id, text, source, topic_tags, linked_metrics).
2. Embeds each chunk using sentence-transformers/all-MiniLM-L6-v2.
3. Stores them in a local ChromaDB collection called "biodiversity_knowledge", preserving all metadata fields so they can be filtered later by topic_tags.
4. Prints a summary of how many chunks were ingested per topic_tag.

Also write retrieval.py with a function `retrieve(query: str, topic_filter: list[str] = None, k: int = 6)` that embeds the query, searches the Chroma collection (optionally filtered by topic_tags), and returns the matched chunks with their metadata and similarity scores.
```

### 6.3 Structured relationship graph prompt
```
Create /knowledge/graph.json containing at least 12 intervention entries following this schema:
{
  "name": string,
  "affects_metrics": [string],
  "mechanism": string (2-3 sentences explaining the causal pathway),
  "effect_size": string (a real, cited quantitative or qualitative estimate),
  "preconditions": [string] (variable states this intervention is suited for, e.g. "low_soil_organic_carbon", "semi_arid", "monoculture"),
  "time_horizon": "short_term" | "medium_term" | "long_term",
  "source": string (real citation, e.g. "FAO", "IPCC AR6 WG2", "IPBES 2019")
}

Cover a spread of interventions across: cover cropping, agroforestry/intercropping, riparian buffer strips, hedgerow/field margin restoration, rotational grazing, wetland restoration, reduced tillage, pollinator strips, integrated pest management, mixed-species reforestation, water harvesting structures, and biochar amendment.

Then write reasoning.py with a function `rank_interventions(user_variables: dict) -> list[dict]` that:
1. Converts the user's raw variable values (e.g., soil_organic_carbon_pct=0.3) into categorical tags (e.g., "low_soil_organic_carbon" if <0.5%, "moderate" if 0.5-1.5%, "high" if >1.5%; similarly bucket rainfall and land use).
2. Scores each intervention in graph.json by counting how many of its `preconditions` match the user's derived tags.
3. Returns interventions sorted by score descending, breaking ties by number of affects_metrics that overlap with variables the user has provided.

This function must require overlap with at least 2 user variables before returning an intervention as a top recommendation, to enforce genuine multi-metric reasoning rather than single-variable lookup.
```

### 6.4 Conversation state machine prompt
```
Write conversation.py implementing a slot-filling conversation manager for a biodiversity advisory chatbot.

Required slots: soil_organic_carbon_pct, soil_ph, soil_moisture, land_use_type, rainfall_pattern, region_climate_zone.
Optional slots: species_richness_observation, pollution_or_deforestation_pressure, geo_coordinates.

Behavior:
1. Maintain a per-session dict: {slots: {}, history: [], last_recommendation: None}.
2. On each user message, call an LLM-based extraction function (I'll wire this to llm_client.py) that returns any slot values it can find in the message, plus the user's implied question/goal.
3. Update the session's slots dict with any newly extracted values (overwrite if the user corrects a previous value).
4. Check which of the 4 mandatory slots (soil_organic_carbon_pct, land_use_type, rainfall_pattern, region_climate_zone) remain empty.
5. If any mandatory slots are still empty, and fewer than 3 total variables are known, return a clarifying question asking for the single most decision-relevant missing slot — prioritize land_use_type and rainfall_pattern before soil_ph/soil_moisture.
6. Once at least 3 environmental variables (mandatory or optional) are known, mark the session ready and hand off to reasoning.py + retrieval.py to generate a recommendation.
7. If the user's new message updates a previously-answered slot, regenerate the recommendation and explicitly state in the reply that it has been revised because of the new information.
```

### 6.5 LLM synthesis prompt template (this is the system prompt your backend sends to the LLM — not something you type once, but the *template* your code assembles per request)
```
You are an AI environmental scientist embedded in a biodiversity advisory system. You are NOT allowed to invent facts, sources, or numbers. You must only use the information provided to you below.

USER'S ENVIRONMENTAL CONTEXT:
{user_variables_as_json}

TOP-RANKED CANDIDATE INTERVENTION (from structured knowledge graph):
{top_intervention_json}

SUPPORTING EVIDENCE (retrieved from knowledge base):
{retrieved_chunks_with_sources}

Using ONLY the information above, produce a response as a single JSON object matching exactly this schema:
{
  "recommendation": string,
  "mechanism": string,
  "metrics_impacted": [{"metric": string, "direction": "increase"|"decrease", "magnitude": string}],
  "time_horizon": string,
  "confidence": "high"|"medium"|"low",
  "source": string,
  "connected_variables": [string]
}

Rules:
- The "mechanism" must explain the causal pathway (why it works), not just restate the recommendation.
- "connected_variables" must list at least 2 of the user's provided variables that jointly justify this recommendation.
- If the evidence is thin, set "confidence" to "medium" or "low" rather than overstating certainty.
- Do not include any text outside the JSON object.
```

### 6.6 Frontend prompt (see Section 7 for the exact design direction to paste in alongside this)
```
Build a clean, minimal single-page chat frontend (plain HTML/CSS/JS, no framework needed unless you prefer React) that:
1. Has a chat panel (message bubbles, user right-aligned, system left-aligned) and, expandable alongside it, a structured input form for the 6 core variables (so a user can either type naturally or fill a form).
2. Renders each recommendation as a distinct card component (not a chat bubble) showing: recommendation title, mechanism (collapsible "why this works"), a small table/list of impacted metrics with direction arrows, a time-horizon badge, a confidence badge, and a source citation line.
3. Follows the visual direction described below [paste Section 7's design spec here].
4. Includes a "Sources used" expandable panel under each response showing which knowledge-base chunks were retrieved (this is important for demoing the RAG pipeline transparently to judges).
5. Has no login/auth — a single session is fine for the hackathon demo.
```

---

## 7. UI/UX Direction — "clean, minimal, official government portal" look

You asked for something that looks like the official biodiversity/environment portals in Switzerland, the UK, Ireland, the USA, and Germany. Here's what those actually share, and how to translate it into your app (these are real, well-documented public design systems and portals you can look at directly for reference):

- **UK — GOV.UK Design System** (design-system.service.gov.uk) and **NBN Atlas / JNCC**: near-black text on white (`#0b0c0c`), a single accent blue for links (`#1d70b8`), no gradients, no drop shadows, generous line-height, one serif-free system font stack, strict left-aligned content, minimal iconography, high color contrast for accessibility, components built around plain rectangular cards with a thin border rather than shadows.
- **USA — USWDS (U.S. Web Design System)** and EPA/USGS biodiversity data portals: similar philosophy — Public Sans font, blue (`#005ea2`) as the single accent, a plain banner strip at the top, boxy card layouts, heavy use of whitespace, data tables kept extremely plain (no zebra striping beyond a faint gray).
- **Ireland — National Biodiversity Data Centre (biodiversityireland.ie)**: white background, green/teal accent (reflecting the ecological subject matter), a simple top nav, large stat counters (e.g., "records / species / datasets") presented as plain numbers with a label underneath — this stat-counter pattern is worth borrowing for your own homepage/dashboard (e.g., "chunks indexed / interventions modeled / variables tracked").
- **Switzerland — admin.ch / BAFU (Federal Office for the Environment)**: red/white restrained palette, very structured grid, sparse typography, almost no decorative imagery — function over form.
- **Germany — UBA (Umweltbundesamt) / BfN**: blue/green restrained palette, information-dense but organized into clear boxed sections, plain sans-serif type, minimal motion/animation.

**Concrete design spec to give your frontend builder:**
- Font: a plain system sans-serif stack (`-apple-system, "Helvetica Neue", Arial, sans-serif`) or Google's "Public Sans"/"Inter" if you want one imported font.
- Palette: white/near-white background, near-black body text, exactly **one** accent color (a muted blue or green — pick one, e.g., `#1d5c3a` for an earthy green or `#1d70b8` for a civic blue) used only for links, buttons, and metric-direction arrows.
- No gradients, no drop shadows — use 1px solid borders (`#e0e0e0`) to separate cards/sections instead.
- Generous whitespace and line-height (1.5–1.6) — this alone reads as "official/trustworthy" more than any other single choice.
- Layout: a plain top bar (project name + a one-line tagline), a two-column body on desktop (chat/input on one side, recommendation cards accumulating on the other), collapsing to a single column on mobile.
- Data presentation: plain tables/lists, no colorful charts unless a chart genuinely clarifies a trend — a simple stat-counter row (like Ireland's site) on your landing/about screen is a strong, easy-to-build touch.
- Micro-detail that reads as "official": a small "Evidence-based · Not a substitute for professional agronomic/ecological advice" disclaimer line in the footer, similar in spirit to how government sites always caveat their guidance.

---

## 8. Day-by-Day Build Plan (excluding CI/CD & deployment, as requested)

**Day 1 — Foundations**
- Run prompt 6.1 (scaffolding).
- Collect and hand-format 20–30 real evidence chunks (FAO, IPCC, IPBES, and a handful of specific peer-reviewed studies) into `/knowledge/documents`.
- Build the structured relationship graph (prompt 6.3) with at least 12 interventions.

**Day 2 — Knowledge layer**
- Run prompt 6.2 (ingestion + retrieval).
- Test retrieval manually: run 5–6 sample queries and confirm relevant chunks come back with correct metadata.
- Build and test `rank_interventions()` against several hand-crafted variable combinations to confirm it genuinely requires ≥2-variable overlap.

**Day 3 — Conversation + reasoning**
- Run prompt 6.4 (conversation state machine).
- Wire retrieval + graph ranking + LLM synthesis (prompt 6.5) into a single `/chat` endpoint and a `/analyze` JSON endpoint (Section 5).
- Validate every LLM response against the pydantic schema; add a retry-on-validation-failure loop.

**Day 4 — Frontend**
- Run prompt 6.6 with the Section 7 design spec pasted in.
- Wire the frontend to both the chat endpoint and the structured form endpoint.
- Add the "Sources used" expandable trace panel.

**Day 5 — Hardening & rubric self-check**
- Walk through the Section 4.3 checklist on at least 5 different test scenarios (vary land use, rainfall, region each time) to make sure recommendations are genuinely non-obvious and multi-variable.
- Write the README (architecture diagram, schema explanation, how retrieval works, how to run locally) — this README doubles as your submission document content (Section 9).
- Record a short demo script covering: a multi-turn chat example, a structured JSON `/analyze` call via Swagger UI, and a look at the sources/graph behind one recommendation.

---

## 9. What your submission document (Word doc) needs to contain

Per the challenge's submission guidelines, prepare a Word document with:
1. GitHub repository link.
2. Live demo URL (only once you deploy — not covered in this blueprint).
3. A README.md overview: architecture, database/schema, local setup, and CI/CD (you said you'll handle CI/CD later — just make sure the README has a placeholder section for it so the document structure matches what's asked).
4. Any credentials/notes needed to run it, plus repo access granted to the four reviewer emails listed in the challenge if your repo is private.

When you're ready to assemble that final Word document, this content can be turned directly into a polished .docx — just say so and it can be built at that point.

---

## 10. What will make you stand out among 800 applicants

- A visibly real, inspectable knowledge graph + vector DB (most entrants will fake this with prompt-only "reasoning").
- Recommendations that explicitly cite which 2+ variables combined to produce them (make this visible in the UI, not just internal logic).
- A "Sources used" trace shown per answer — judges can verify grounding in seconds, which builds trust fast.
- The GOV.UK/USWDS-style minimal UI — will visually read as "serious/scientific" next to typical hackathon chat-bubble UIs.
- A structured `/analyze` JSON endpoint with auto-generated Swagger docs — shows engineering maturity beyond just a chatbot.
- Handling variable *updates* mid-conversation gracefully (revising a recommendation live) — most basic chatbots don't handle this and it's an easy, high-visibility "conversational intelligence" win.
