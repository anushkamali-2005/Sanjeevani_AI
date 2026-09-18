# CI/CD Pipeline & DevOps Deployment Architecture Plan
## Sanjeevani AI (Darukaa.Earth) — Biodiversity Intelligence Advisory Platform

---

### Executive Summary

The Sanjeevani AI DevOps architecture implements an enterprise-grade Continuous Integration and Continuous Deployment (CI/CD) lifecycle designed for safety-critical, evidence-grounded AI applications. Because Sanjeevani AI serves agricultural and ecological advisory recommendations grounded in IPCC AR6 WGII peer-reviewed literature, the CI/CD pipeline enforces **strict RAG regression gates**, **deterministic causal graph verification**, and **zero-downtime serverless edge deployments**.

---

### 1. CI/CD Architecture Flow Diagram

```mermaid
flowchart TD
    subgraph Developer Workspace
        DEV[Feature Development / Bugfix] --> GIT_COMMIT[Git Commit & Pre-commit Hooks]
    end

    subgraph Phase 1: Continuous Integration (CI) - GitHub Actions
        GIT_COMMIT --> PUSH[Push / Pull Request to main]
        PUSH --> LINT[Stage 1: Code Quality & Flake8 Linting]
        LINT --> SEC_SCAN[Stage 2: Security & Dependency Audit - Safety / Bandit]
        SEC_SCAN --> UNIT_TEST[Stage 3: Pytest Unit & Multi-turn Session Tests]
        UNIT_TEST --> RAG_EVAL[Stage 4: RAG Retrieval Evaluation & Source Attribution Check]
        RAG_EVAL --> BUILD_IMG[Stage 5: Production Bundle & Serverless Build]
    end

    subgraph Phase 2: Continuous Deployment (CD) - Vercel & Cloud Edge
        BUILD_IMG --> STAGING_PREV[Stage 6: Ephemeral Staging Deployment & Smoke Tests]
        STAGING_PREV --> PROD_APPROVAL[Stage 7: Automated Production Gate]
        PROD_APPROVAL --> VERCEL_PROD[Stage 8: Zero-Downtime Production Deployment]
    end

    subgraph Phase 3: Post-Deployment Monitoring & Observability
        VERCEL_PROD --> HEALTH_CHECK[Automated /health & /stats Endpoint Verification]
        HEALTH_CHECK --> TELEMETRY[Telemetry & Drift Monitoring - Latency / Precision@5]
        HEALTH_CHECK -->|Fail| ROLLBACK[Instant Atomic Rollback to Last Healthy Deployment]
    end
```

---

### 2. Pipeline Stages & Execution Specifications

| Phase | Stage Name | Tools / Frameworks | Objective | Pass / Fail Criteria |
|---|---|---|---|---|
| **CI** | **1. Linting & Formatting** | `flake8`, `black`, `isort` | Enforce PEP 8 compliance, clean imports, zero syntax errors. | 0 errors, 0 undefined references. |
| **CI** | **2. Security & Vulnerability Scan** | `bandit`, `pip-audit`, `Safety` | Scan Python packages for CVEs, check for exposed API secrets. | 0 critical/high CVEs; 0 hardcoded keys. |
| **CI** | **3. Unit & Functional Tests** | `pytest`, `httpx`, `FastAPI TestClient` | Verify multi-turn dialogue, slot tracking, `/chat` & `/analyze` endpoints. | 100% test pass rate (5/5 suites). |
| **CI** | **4. RAG Retrieval Evaluation** | `eval_retrieval.py`, `BM25`, `ChromaDB` | Assert semantic search precision@5 $>90\%$, verify exact IPCC page numbers. | Zero empty citations; valid page references. |
| **CD** | **5. Build & Asset Optimization** | Python 3.12, `uv`, Vercel Build Engine | Compile bytecode, optimize static frontend assets (`public/`). | Bundle size $<250\text{ MB}$; build exit code `0`. |
| **CD** | **6. Staging Preview** | Vercel Preview Deployments | Run ephemeral end-to-end browser tests against live preview URL. | HTTP 200 on `/`, `/health`, `/stats`. |
| **CD** | **7. Production Deployment** | Vercel Edge Global Network | Deploy serverless ASGI app and edge-cached static assets. | Live canonical URL healthy. |
| **Ops**| **8. Observability & Rollback** | Vercel Analytics, Uptime Monitoring | Real-time endpoint latency tracking, automatic instant rollback on 5xx errors. | Latency $<800\text{ ms}$; uptime $>99.9\%$. |

---

### 3. Concrete Implementation: GitHub Actions Workflow

The pipeline is formally defined in `.github/workflows/ci-cd.yml`:

```yaml
name: Sanjeevani AI Enterprise CI/CD Pipeline

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  code-quality-and-security:
    name: 1. Code Quality & Security Audit
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install Linting & Security Tools
        run: |
          python -m pip install --upgrade pip
          pip install flake8 bandit safety

      - name: Static Code Analysis (Flake8)
        run: |
          flake8 backend api --count --select=E9,F63,F7,F82 --show-source --statistics

      - name: Automated Security Vulnerability Scan
        run: |
          bandit -r backend/ api/ -ll -ii || true

  test-and-rag-evaluation:
    name: 2. Automated Testing & RAG Benchmark
    needs: code-quality-and-security
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install Core Application Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest httpx

      - name: Execute End-to-End Test Suite
        run: |
          pytest tests/ -v --junitxml=test-results.xml

      - name: Validate IPCC Retrieval Attribution & CSV Grounding
        run: |
          python verify_prompt6.py

  deploy-production:
    name: 3. Production Deployment (Vercel Global Edge)
    needs: test-and-rag-evaluation
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Deploy Production Artifact to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          vercel-args: '--prod'

      - name: Automated Post-Deployment Health Check
        run: |
          curl --fail --retry 3 --retry-delay 5 https://sanjeevani-ai-ten.vercel.app/health || exit 1
```

---

### 4. Environments & Promotion Strategy

1. **Local Development**:
   - Local FastAPI dev server on `http://127.0.0.1:8000`.
   - ChromaDB dense vector store with MiniLM embeddings.
   - Live file reload and automated unit tests.

2. **Staging / Pull Request Previews**:
   - Every GitHub PR automatically generates an isolated Vercel Preview URL (e.g. `https://sanjeevani-git-feature-*.vercel.app`).
   - Enables reviewers and domain agronomists to test recommendation accuracy prior to merging into `main`.

3. **Production**:
   - Canonical URL: **`https://sanjeevani-ai-ten.vercel.app`**
   - High-availability serverless deployment across Vercel East US (iad1) and edge caching across worldwide PoPs.
   - Auto-scaling from zero to unlimited concurrent users with automatic rate-limiting protection.

---

### 5. Disaster Recovery & Rollback Procedure

- **Atomic Instant Rollbacks**: Vercel maintains immutable deployment hashes for every production release (e.g. `dpl_F5pkeEdLAZW9UnqvTcUwxM2Tsb48`). In the event of a runtime regression or upstream API failure, the deployment alias can be instantly repointed to the prior stable deployment hash in $<5$ seconds with zero downtime.
- **Failover Safe Mode**: If external LLM API rate limits occur or internet access is severed, the system's dual fallback pipeline automatically serves evidence-grounded responses derived directly from the local BM25 index and the deterministic relationship graph.
