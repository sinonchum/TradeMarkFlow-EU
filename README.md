# TradeMarkFlow-EU

**Privacy-first AI trademark analysis & clearance system for the European Union**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)](https://fastapi.tiangolo.com)
[![LanceDB](https://img.shields.io/badge/LanceDB-0.25+-orange)](https://lancedb.com)
[![Agno Agent](https://img.shields.io/badge/Agno-2.5+-purple)](https://github.com/agno-agi/agno)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Intelligent trademark clearance search, risk assessment, and Office Action response drafting — powered by multilingual semantic search and legal-aware AI agents.

---

## 🚀 What is TradeMarkFlow-EU?

TradeMarkFlow-EU is an open-source **legal technology (LegalTech) framework** for automating European Union trademark (EUTM) analysis workflows. It combines **RAG-powered semantic search** with **multi-agent orchestration** to deliver:

- **Trademark clearance search** across 24 EU languages
- **Similarity risk assessment** (phonetic, visual, and semantic dimensions)
- **Office Action parsing and response drafting** for EUIPO examination reports
- **Nice Classification recommendation** from business descriptions

Built for **trademark attorneys, IP professionals, and legal tech teams** who need reliable, explainable AI assistance without compromising client confidentiality.

---

## ✨ Key Features

### 🔒 Privacy-First Architecture
- **Local LLM support** via [Ollama](https://ollama.com) — process sensitive brand concepts entirely offline
- **On-device vectorization** — no data leaves your infrastructure for embeddings
- **Configurable cloud/local routing** — switch between OpenAI and Ollama via environment variable

### 🌍 True Multilingual Semantic Search
- Powered by [**multilingual-e5-large**](https://huggingface.co/intfloat/multilingual-e5-large) (1024-dim) trained on all 24 official EU languages
- Cross-lingual retrieval: query in Chinese, find matches in German, French, Italian…
- No translation layer needed — semantically equivalent terms align in vector space

### ⚡ Hybrid SQL + Vector Search
- [**LanceDB**](https://lancedb.com) serverless vector database — zero infrastructure, single-file deployment
- Combine structured filters (Nice Class, status, priority date) with semantic similarity
- Sub-millisecond retrieval on indexed trademark corpora

### 🤖 Multi-Agent Legal AI
- **Clearance Agent**: end-to-end trademark clearance with risk scoring
- **OA Response Agent**: parse EUIPO Office Actions, cite EUTMR articles, draft legal arguments
- Built on [**Agno**](https://github.com/agno-agi/agno) agent framework with tool-calling and conversation memory

### 📄 Document Intelligence
- PDF Office Action parsing via [**Marker**](https://github.com/VikParuchuri/marker) — high-fidelity markdown conversion
- Web scraping via [**Scrapling**](https://github.com/foongminwong/scrapling) — adaptive, stealthy EUIPO bulletin monitoring

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Gateway                       │
│              (REST/WebSocket endpoints)                  │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │Clearance │  │   OA     │  │  Search  │
  │  Agent   │  │Response  │  │  Agent   │
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │             │             │
       ▼             ▼             ▼
┌─────────────────────────────────────────────────────────┐
│                  Tool & Adapter Layer                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ LanceDB  │ │EUIPO     │ │TMclass   │ │LightRAG  │   │
│  │  Tools   │ │Adapter   │ │ Adapter  │ │ Legal DB │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │
└───────┼────────────┼────────────┼────────────┼──────────┘
        ▼            ▼            ▼            ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ LanceDB  │ │ EUIPO    │ │  WIPO    │ │  Legal   │
  │(Vectors) │ │ eSearch  │ │ APIs     │ │ Corpus   │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘

        ┌──────────────────────────────┐
        │  Embedding Service           │
        │  multilingual-e5-large (CPU) │
        └──────────────────────────────┘
```

> Full Mermaid diagram available in [`docs/architecture.md`](docs/architecture.md)

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Agent Framework** | [Agno](https://github.com/agno-agi/agno) | Multi-agent orchestration, tool-calling, memory |
| **API Layer** | [FastAPI](https://fastapi.tiangolo.com) | Async REST/WebSocket endpoints |
| **Vector Database** | [LanceDB](https://lancedb.com) | Serverless hybrid SQL + vector search |
| **Embeddings** | [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | 1024-dim multilingual semantic vectors |
| **Web Scraping** | [Scrapling](https://github.com/foongminwong/scrapling) | Adaptive, stealthy crawling |
| **PDF Parsing** | [Marker](https://github.com/VikParuchuri/marker) | PDF → Markdown conversion |
| **Local LLM** | [Ollama](https://ollama.com) | Privacy-sensitive inference |
| **Cloud LLM** | OpenAI GPT-4o | High-performance general inference |
| **Legal RAG** | [LightRAG](https://github.com/HKUDS/LightRAG) | Legal corpus knowledge graph |

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/sinonchum/TradeMarkFlow-EU.git
cd trademarkflow-eu

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your API keys (or set DEFAULT_LLM=local for Ollama)
```

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...           # For cloud inference
OLLAMA_HOST=http://localhost:11434  # For local inference
DEFAULT_LLM=cloud               # cloud | local

# Model Selection
OPENAI_MODEL_ID=gpt-4o-mini
LOCAL_MODEL_ID=qwen2.5:14b

# Data Paths
LANCEDB_PATH=./data/lancedb/eu_trademarks.lance
LEGAL_CORPUS_PATH=./legal_corpus/
```

---

## 🚦 Quick Start

### 1. Initialize Vector Database

```python
from app.models.schema import create_trademark_table

table = create_trademark_table("./data/lancedb/eu_trademarks.lance")
print(f"Table created: {table}")
```

### 2. Run a Clearance Search

```python
from app.agents.agents import run_clearance

report = run_clearance(
    mark_name="SolarFlux",
    business_description="Solar panel installation and maintenance services",
)
print(report)
```

### 3. Start the API Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API documentation available at `http://localhost:8000/docs`

---

## 📖 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/clearance` | Run trademark clearance search |
| `POST` | `/api/v1/oa-response` | Parse Office Action & draft response |
| `GET` | `/api/v1/trademarks/search` | Hybrid vector + SQL search |
| `GET` | `/api/v1/trademarks/{app_number}` | Lookup by application number |
| `GET` | `/api/v1/stats` | Database statistics |
| `POST` | `/api/v1/ingest` | Trigger EUIPO data ingestion |

---

## 🔬 How It Works

### Clearance Search Pipeline

```
User Input: "SolarFlux" + "solar energy services"
    │
    ▼
Step 1: TMclass API → Recommend Nice Classes [9, 35, 37, 42]
    │
    ▼
Step 2: LanceDB Hybrid Search
    ├── SQL filter: WHERE nice_class IN (9, 35, 37, 42)
    └── Vector similarity: query:"SolarFlux solar energy" → top-10
    │
    ▼
Step 3: 3D Similarity Analysis
    ├── Phonetic: "SolarFlux" vs "Solflux" vs "SunFlux"
    ├── Visual:   logo shape/wordmark pattern matching
    └── Semantic: embedding distance across languages
    │
    ▼
Step 4: Absolute Grounds Check (EUTMR Art. 7)
    └── RAG retrieval from legal corpus
    │
    ▼
Output: Risk Assessment Report (Low/Medium/High/Critical)
```

### Office Action Response Pipeline

```
Input: EUIPO Office Action PDF
    │
    ▼
Marker PDF → Markdown conversion
    │
    ▼
AI Parser:
    ├── Classify each objection (Art. 7 absolute / Art. 8 relative)
    ├── Extract cited EUTMR articles and guidelines
    └── Identify precedent case references
    │
    ▼
Legal RAG:
    ├── Retrieve relevant EUTMR articles from knowledge base
    └── Find Board of Appeal decisions for similar cases
    │
    ▼
Response Agent:
    └── Draft professional legal arguments per objection
    │
    ▼
Output: Structured response draft with citations
```

---

## 📊 Data Schema

Each trademark record in LanceDB contains:

| Field | Type | Description |
|-------|------|-------------|
| `application_number` | `str` | EUIPO application ID |
| `mark_name` | `str` | Trademark name |
| `mark_type` | `str` | `Word`, `Figurative`, `Combined` |
| `nice_classes` | `list[int]` | Nice Classification classes |
| `goods_services` | `str` | Goods/services description |
| `status` | `str` | `Registered`, `Pending`, `Refused` |
| `applicant_name` | `str` | Applicant/owner |
| `filing_date` | `str` | ISO 8601 date |
| `priority_date` | `str` | Priority claim date (if any) |
| `embedding` | `vector[1024]` | multilingual-e5-large vector |

---

## 🌐 EUIPO Multilingual Data Strategy

TradeMarkFlow-EU handles the 24 official EU languages through:

1. **Unified embedding space**: `multilingual-e5-large` maps all languages to a shared 1024-dimensional vector space — semantically equivalent terms cluster regardless of language
2. **Prefixed embedding**: `query:` prefix for search, `passage:` prefix for indexing (E5 model requirement)
3. **Hybrid filtering**: Structured fields (Nice Class, status) combined with semantic vector similarity in a single LanceDB query
4. **Original text preservation**: Source language retained for legal citation accuracy

---

## 🗺️ Roadmap

- [ ] **v0.2** — EUIPO bulletin auto-monitoring via Scrapling
- [ ] **v0.3** — CLIP integration for figurative/logo similarity search
- [ ] **v0.4** — WIPO Madrid Protocol support (international registrations)
- [ ] **v0.5** — LightRAG legal knowledge graph with EUTMR + BoA decisions
- [ ] **v0.6** — Telegram/Lark bot integration for on-the-go alerts
- [ ] **v1.0** — Production-ready deployment with Docker Compose

---

## 🤝 Contributing

Contributions welcome! This project follows [semantic versioning](https://semver.org) and [conventional commits](https://www.conventionalcommits.org).

```bash
# Development setup
git clone https://github.com/sinonchum/TradeMarkFlow-EU.git
cd trademarkflow-eu
pip install -e ".[dev]"
pytest tests/
```

### Areas needing help

- 🌍 **Multilingual test corpus**: sample trademarks in all 24 EU languages
- ⚖️ **Legal validation**: verify EUTMR article citations and BoA case references
- 🕷️ **EUIPO scraper resilience**: adaptive selectors for eSearch portal changes
- 📊 **Benchmark datasets**: clearance search accuracy metrics

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🔗 Related Projects

- [PatentFlow](https://github.com/sinonchum/PatentFlow) — Privacy-first patent analysis (sister project)
- [Agno](https://github.com/agno-agi/agno) — Multi-agent framework
- [LanceDB](https://github.com/lancedb/lancedb) — Serverless vector database

---

## 📚 Keywords

`trademark clearance` `EU trademark search` `EUTM` `EUIPO` `Nice classification` `legal AI` `legal tech` `RAG` `multilingual search` `vector database` `trademark similarity` `Office Action response` `intellectual property` `IP automation` `privacy-first AI` `LanceDB` `semantic search` `trademark risk assessment`
