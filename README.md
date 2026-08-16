# ECG Clinical Intelligence Pipeline

Real-time / offline ECG monitoring and clinical diagnosis system that combines:

| Branch | What it does |
|---|---|
| **CNN branch** | Beat-level classification into 5 AAMI classes (**N / S / V / F / Q**) by a trained convolutional network (`ecg_cnn_model.keras`) |
| **Clinical branch** | NeuroKit2 morphology extraction — PR / QRS / QT / ST intervals, P / T amplitudes entering features (`neurokit_feature_extractor.py`) |
| **Temporal analysis** | Sliding-window rhythm detection — bigeminy, trigeminy, couplets, pauses, AV block, VT / VF, rates |
| **Disease detection** | Rule-based clinical detectors for 15+ conditions using AHA / ESC thresholds (`disease_detection/`) |
| **RAG agent** | Semantic retrieval over a clinical PDF knowledge base (`Cardiac_Diagnostics_Comprehensive_KB.pdf`) + LLM-generated assessment (Groq API) |

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Environment variables](#3-environment-variables)
4. [MIT-BIH data setup](#4-mit-bih-data-setup)
5. [Run the system](#5-run-the-system)
6. [RAG knowledge base (PDF)](#6-rag-knowledge-base-pdf)
7. [Kafka streaming (optional)](#7-kafka-streaming-optional)
8. [Offline dataset export](#8-offline-dataset-export)
9. [Tests](#9-tests)
10. [Notebooks](#10-notebooks)
11. [Project structure](#11-project-structure)

---

## 1. Requirements

- Python **3.10+** (developed on 3.13)
- PostgreSQL **14+** with the [pgvector](https://github.com/pgvector/pgvector) extension
- Optional: Apache Kafka broker on `localhost:9092`
- A local copy of the **MIT-BIH arrhythmia database** (see [§4](#4-mit-bih-data-setup))

---

## 2. Installation

```bash
git clone https://github.com/MORO-66t/Agntic-rrag-system-ECG-based-clinical-diagnose.git
cd <repo-directory>

python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**Database (PostgreSQL):** the schema — including the `vector` embedding column — is created
automatically on the first run. One-time setup:

```bash
createdb ecg_agent_data
psql -d ecg_agent_data -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Adjust host / port / name / user / password in `.env` if needed.

**Secrets:** copy the template and fill in your real credentials. **Never commit `.env`**
(it is gitignored):

```bash
# Linux / macOS
cp .env.example .env
# Windows
copy .env.example .env
```

---

## 3. Environment variables

All configuration is read from `.env` by `config.py`. Full list is in `.env.example`:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Single Groq API key |
| `GROQ_API_KEYS` | Comma-separated Groq keys (rotation; overrides `GROQ_API_KEY`) |
| `GROQ_MODEL` | Groq chat model, e.g. `openai/gpt-oss-20b` |
| `LLM_MODE` | `groq` (default) \| `test` \| `fake` |
| `LOCAL_LLM_MODEL_PATH` | Folder of a local model for `test` mode |
| `HF_TOKEN` / `HF_TOKENS` | Optional HuggingFace tokens (used by `token_manager.py`) |
| `DB_HOST` `DB_PORT` `DB_NAME` `DB_USER` `DB_PASSWORD` | PostgreSQL connection |
| `PDF_DOCUMENT_NAME` `PDF_DEFAULT_PATH` | Clinical knowledge PDF |
| `KAFKA_BOOTSTRAP_SERVERS` and `KAFKA_*_TOPIC` | Optional Kafka brokers / topics |

---

## 4. MIT-BIH data setup

The pipeline loads WFDB records directly from local disk (no pre-generated CSV needed).
Place the PhysioNet MIT-BIH arrhythmia database and the optional P/T landmark `.mat`
files under `tr0ph1c/` (this folder is gitignored — set it up locally):

```text
tr0ph1c/mit-bih-arrhythmia-dataset-lead-ii/versions/2/
├── mit-bih-arrhythmia-database-1.0.0/                          # .dat/.hea/.atr records
└── Pwaves_Twaves_Annotation/Pwaves_Twaves_Annotation/          # optional Ant_mitdb_XXX.mat
```

Missing `.mat` landmarks only disable P/T annotation validation — the pipeline still runs.

---

## 5. Run the system

The main entry point is `realtime_stream.py`: it simulates a real-time bedside monitor by
feeding each beat of a record into the dual-branch pipeline (CNN + NeuroKit2 + temporal
analysis + disease detection + RAG agent).

### Quick start

```bash
# Fast simulation of record 100 (agent ON, no inter-beat delay)
python realtime_stream.py --record 100 --fast

# Multiple records in one run
python realtime_stream.py --record 100 --record 208 --fast

# True real-time pacing (sleeps by each beat's RR interval)
python realtime_stream.py --record 100

# Process a subset and write a JSON diagnostics report
python realtime_stream.py --record 208 --start-beat 0 --end-beat 2000 --report out.json --fast

# Dry run — sanity-check the data source only, no pipeline calls
python realtime_stream.py --record 100 --dry-run --fast
```

### All CLI flags (`realtime_stream.py`)

| Flag | Description |
|---|---|
| `--record <NAME>` | MIT-BIH record name (e.g. `100`, `208`). Repeatable to stream multiple records |
| `--max-beats N` | Stop after N beats |
| `--start-beat N` / `--end-beat N` | 0-based beat index range (end inclusive) |
| `--start-time S` / `--end-time S` | Window over the record clock (seconds) |
| `--fast` | No inter-beat delay (default behaviour) |
| `--realtime` | Sleep by each beat's RR interval |
| `--dry-run` | Skip pipeline calls; only trace the data source |
| `--session-id STR` | Session id stored in PostgreSQL |
| `--model-path FILE` | Path to the CNN weights (default `ecg_cnn_model.keras`) |
| `--agent-llm-mode {groq,test}` | RAG agent LLM backend (`groq` default, `test` = local model) |
| `--local-model-path DIR` | Local model directory (with `--agent-llm-mode test`) |
| `--no-agent` | Disable the RAG agent entirely |
| `--report PATH` | Write a JSON diagnostics report |
| `--quiet-events` | Suppress per-event diagnostic logging |
| `--turbo` | Fast-test mode (no agent, minimal thresholds) |
| `--debug-afib` | Print the AFib diagnostic dashboard for every analyzed window |
| `--debug-afl` | Print the Atrial Flutter diagnostic dashboard for every analyzed window |
| `--debug-morphology` | Show exactly how each ECG feature was computed (per beat) |
| `--log-file PATH` | Write all log output to a file in addition to the terminal |

> **Note:** `--agent-llm-mode inference` was removed — the HuggingFace inference backend is
> gone. Use `groq` (requires `GROQ_API_KEYS`) or `test` (requires a local model).

---

## 6. RAG knowledge base (PDF)

Embedding uses the local `sentence-transformers/all-MiniLM-L6-v2` model (first run
downloads it from the HuggingFace Hub). Query generation via `--analyze-event` uses Groq,
so `GROQ_API_KEYS` must be configured in `.env`.

```bash
# Extract, embed, and store the PDF chunks in PostgreSQL
python pdf_semantic_rag.py --ingest

# Shortcut for the default PDF
python ingest_pdf_knowledge.py

# Semantic search over the knowledge base
python pdf_semantic_rag.py --search "long QT criteria"

# Full RAG agent analysis for an event type (LLM generation)
python pdf_semantic_rag.py --analyze-event "Atrial Fibrillation (AF)" \
    --session-id demo --top-k 5

# Custom PDF / document name
python pdf_semantic_rag.py --ingest --pdf docs/clinical_kb.pdf --document-name "My KB"
```

---

## 7. Kafka streaming (optional)

Streams raw ECG through a Kafka broker instead of calling the pipeline directly.

### 7.1 Stack and versions

| Component | Version used / recommended | Notes |
|---|---|---|
| `kafka-python` | `>=2.0.2` (declared in `requirements.txt`) | Pure-Python client used by all `kafka_*.py` modules |
| Apache Kafka (broker) | **3.4.x** (recommended) | Any Kafka 3.x works; kafka-python 2.0.2 is protocol-compatible |
| Java (JVM) | **11 or 17 (LTS)** | Required to run the Kafka broker scripts (`bin/kafka-*.sh`) |
| ZooKeeper | Bundled with the Kafka distribution | Only needed in classic (non-KRaft) broker mode |
| Python | **3.10+** (developed on 3.13) | kafka-python 2.0.2 supports Python 3.6+ |

### 7.2 Install a Kafka broker (one-time)

Download an Apache Kafka 3.4.x distribution and a matching JDK (11/17), or run the Docker image below (no install):

```bash
# Native (Linux/macOS) — Apache Kafka 3.4.0 (Scala 2.13)
wget https://downloads.apache.org/kafka/3.4.0/kafka_2.13-3.4.0.tgz
tar -xvzf kafka_2.13-3.4.0.tgz
cd kafka_2.13-3.4.0

# Verify Java (Kafka 3.4 supports Java 11 and 17)
java -version
```

Docker alternative (single broker, no install):

```bash
docker run -d --name ecg-kafka -p 9092:9092 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_PROCESS_ROLES=broker,controller \
  -e KAFKA_NODE_ID=1 \
  -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 \
  -e KAFKA_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093 \
  -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  apache/kafka:3.7.0
```

### 7.3 Start ZooKeeper + Kafka broker (classic mode)

**Terminal 1 — ZooKeeper:**

```bash
cd kafka_2.13-3.4.0
bin/zookeeper-server-start.sh config/zookeeper.properties
```

**Terminal 2 — Kafka broker:**

```bash
cd kafka_2.13-3.4.0
bin/kafka-server-start.sh config/server.properties
```

> **KRaft mode** (no ZooKeeper, Kafka 3.x+):
> ```bash
> bin/kafka-storage.sh random-uuid                      # prints a UUID
> bin/kafka-storage.sh format -t <UUID> -c config/kraft/server.properties
> bin/kafka-server-start.sh config/kraft/server.properties
> ```

### 7.4 Create the 4 topics

Run once while the broker is up:

```bash
cd kafka_2.13-3.4.0

bin/kafka-topics.sh --bootstrap-server localhost:9092 --create \
    --topic ecg.raw.signal            --partitions 1 --replication-factor 1
bin/kafka-topics.sh --bootstrap-server localhost:9092 --create \
    --topic ecg.events.temporal       --partitions 1 --replication-factor 1
bin/kafka-topics.sh --bootstrap-server localhost:9092 --create \
    --topic ecg.results.clinical      --partitions 1 --replication-factor 1
bin/kafka-topics.sh --bootstrap-server localhost:9092 --create \
    --topic ecg.patient.registration  --partitions 1 --replication-factor 1

# Verify all four topics exist
bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
```

### 7.5 Run the pipeline over Kafka

**Step 1 — start the processing service (Terminal 3):**

```bash
python ecg_kafka_service.py
```

> The service consumes with `auto_offset_reset="latest"` — start it **before** the
> producer, otherwise the messages produced earlier are skipped.

**Step 2 — stream a MIT-BIH record (Terminal 4):**

```bash
# Stream raw chunks to ecg.raw.signal (~1 chunk = 1 second @ 360 Hz)
python -c "from kafka_mitbih_producer import stream_mitbih_to_kafka; stream_mitbih_to_kafka('202', max_chunks=600)"
```

Or run the full end-to-end test (starts the service in a thread, streams, and verifies the output topics):

```bash
python test_kafka_full_pipeline.py                     # record 202, 5 beats
python test_kafka_full_pipeline.py --record 100 --max-beats 20
python test_kafka_full_pipeline.py --no-service        # if the service is already running
```

Quick producer smoke test without a broker:

```bash
python test_mitbih_producer.py
```

**Step 3 — inspect the output topics (Terminal 5):**

```bash
bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 \
    --topic ecg.events.temporal --from-beginning
bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 \
    --topic ecg.results.clinical --from-beginning
```

Topics (configurable via `KAFKA_*` in `.env`):

| Topic | Stream | Written by | Read by |
|---|---|---|---|
| `ecg.raw.signal` | 1 — raw ECG chunks | `kafka_mitbih_producer.py` / `test_kafka_full_pipeline.py` | `ecg_kafka_service.py` |
| `ecg.events.temporal` | 2 — rhythm/pattern events | `ecg_kafka_service.py` | your app / test verifier |
| `ecg.results.clinical` | 3 — clinical + agent results | `ecg_kafka_service.py` | your app / test verifier |
| `ecg.patient.registration` | 4 — patient metadata | `test_kafka_full_pipeline.py` | `ecg_kafka_service.py` |

Relevant modules: `kafka_mitbih_producer.py`, `kafka_raw_consumer.py`, `kafka_producer.py`, `ecg_kafka_service.py`.

---

## 8. Offline dataset export

Build a flat CSV of beats + features for training / evaluation (a separate use case from
the live pipeline):

```bash
python convert_wfdb_to_csv_W.py     # writes ./mitbih_beats1_with_pt.csv
```

---

## 9. Tests

```bash
python test_mitbih_producer.py       # producer chunk smoke test
python test_kafka_full_pipeline.py   # Kafka end-to-end (requires a broker)
```

---

## 10. Notebooks

| Notebook | Purpose |
|---|---|
| `models-training.ipynb` | Trains the CNN classifier → save as `ecg_cnn_model.keras` |
| `ecg-project1.ipynb` | Original experiment / EDA |
| `mit_bih_neurokit2_analysis.ipynb` | NeuroKit2 morphology exploration |
| `mit_bih_neurokit2_analysis_executed.ipynb` | Same, with cell outputs |
| `neurokit_morphology_visualization.ipynb` | Morphology feature visualization |

---

## 11. Project structure

```text
realtime_stream.py             entry point — simulated real-time monitor
ecg_pipeline.py                dual-branch orchestrator (CNN + clinical)
convert_wfdb_to_csv_W.py       beat segmentation + CNN preprocessing
neurokit_feature_extractor.py  NeuroKit2 morphology extraction
feature_engineering.py         feature assembly
clinical_report_formatter.py   clinical report formatting
temporal_analysis.py           rhythm / episode analysis
disease_detection/             rule-based disease detectors
disease_detector.py            detector orchestrator
Event_Manager.py               event rules / escalation
episode_manager.py             episode lifecycle
pdf_semantic_rag.py            RAG ingestion / retrieval / agent
llm_client.py                  LLM client factory (Groq / local / fake)
database.py                    PostgreSQL + pgvector persistence
kafka_*.py, ecg_kafka_service.py   optional Kafka streaming
token_manager.py               HuggingFace token rotation utility
```

> **Note:** the dataset folder (`tr0ph1c/`), runtime logs, and generated artifacts are
> all gitignored — check `.gitignore` and set the data up locally ([§4](#4-mit-bih-data-setup)).

