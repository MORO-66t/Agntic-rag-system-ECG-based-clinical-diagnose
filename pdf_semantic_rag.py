"""
Semantic-only PostgreSQL pgvector RAG over Cardiac_Diagnostics_Comprehensive_KB.pdf.

This module intentionally does not use the old RAG/event-query files. The runtime
path is:

    PDF -> text chunks -> sentence-transformer embeddings -> PostgreSQL pgvector
        -> semantic retrieval -> clinical prompt -> LLM -> stored interaction

The default embedding model is all-MiniLM-L6-v2 because the existing database
uses VECTOR(384). If you later change the table dimension, change
EMBEDDING_MODEL_NAME and the database vector dimension together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import numpy as np
import pypdf

from database import ECGDatabase
from llm_client import build_llm_client
from clinical_report_formatter import render_report_or_fallback_en, render_patient_summary_en
from llm_report_renderer import render_via_llm, render_patient_report_via_llm

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = BASE_DIR / "Cardiac_Diagnostics_Comprehensive_KB.pdf"
DEFAULT_DOCUMENT_NAME = "Cardiac_Diagnostics_Comprehensive_KB.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_CHUNK_WORDS = 430
CHUNK_OVERLAP_WORDS = 70
DEFAULT_TOP_K = 5


_EMBEDDING_MODEL: Optional["SentenceTransformer"] = None


def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def words(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def infer_section_hint(text: str) -> str:
    candidates = [
        "Definition",
        "Mechanism",
        "ECG Findings",
        "Required ECG Features",
        "Diagnostic Criteria",
        "Differential Diagnosis",
        "Symptoms",
        "Risk Factors",
        "Immediate Actions",
        "Treatment Overview",
        "Follow-up Questions",
        "References",
        "Knowledge Base Architecture",
        "Chunking Strategy",
        "Embedding Recommendations",
    ]
    lowered = text.lower()
    for candidate in candidates:
        if candidate.lower() in lowered[:500]:
            return candidate
    match = re.search(
        r"([A-Z][A-Za-z0-9 /()\-]{3,80})\s+Category:\s+",
        text
    )
    if match:
        return match.group(1).strip()
    return "general"


def stable_chunk_id(document_name: str, page_start: int, page_end: int, chunk_index: int, content: str) -> str:
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"{document_name}:p{page_start}-{page_end}:c{chunk_index}:{digest}"


@dataclass
class PDFTextChunk:
    document_name: str
    chunk_id: str
    page_start: int
    page_end: int
    chunk_index: int
    section_hint: str
    content: str
    metadata_json: Dict[str, Any]

    def to_insert_dict(self, embedding: Iterable[float]) -> Dict[str, Any]:
        return {
            "document_name": self.document_name,
            "chunk_id": self.chunk_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_index": self.chunk_index,
            "section_hint": self.section_hint,
            "content": self.content,
            "metadata_json": self.metadata_json,
            "embedding": list(map(float, embedding)),
        }


class PDFSemanticChunker:
    # This document has a fixed 2-line running header on EVERY page
    # ("2. Knowledge Base Architecture" + a bare page number) that is a
    # print-layout artifact, not content — verified across pages 1, 51,
    # 121, 201, and 261 of this specific KB, all identical regardless of
    # which section they're actually in. Stripped so it doesn't pollute
    # chunk content or confuse entry-boundary detection below.
    _RUNNING_HEADER_RE = re.compile(r"^2\.\s*Knowledge Base Architecture$")
    _PAGE_NUMBER_LINE_RE = re.compile(r"^\d+$")

    # Every one of the 57 disease entries in this KB begins with its own
    # title on one line, immediately followed by a line of the form
    # "Category: <category> Priority: <CRITICAL|HIGH|MODERATE> ICD-10: <code>".
    # Verified this pattern matches exactly 57 times in this document, with
    # a clean, unambiguous title line immediately preceding it every time —
    # this is what makes tagging "which disease does this chunk belong to"
    # possible without maintaining a separate hardcoded disease list.
    _ENTRY_HEADER_RE = re.compile(
        r"^Category:\s*(?P<category>.+?)\s+Priority:\s*(?P<priority>\w+)\s+ICD-10:\s*(?P<icd10>[\w.\-]+)\s*$"
    )

    def __init__(
        self,
        max_words: int = MAX_CHUNK_WORDS,
        overlap_words: int = CHUNK_OVERLAP_WORDS
    ):
        self.max_words = max_words
        self.overlap_words = overlap_words

    @classmethod
    def _strip_running_header(cls, raw_text: str) -> str:
        """
        Remove the fixed running header + bare page number when they are the
        first two lines of the page, in EITHER order — this document is not
        consistent about which comes first (verified: pages 24-28 render
        "2. Knowledge Base Architecture" then the page number; page 7 renders
        the page number then that same header text). Only strips at the
        very start of the page, so a legitimate later occurrence of this
        phrase (e.g. the real Table of Contents entry, or the real
        "Knowledge Base Architecture References" appendix title at the end
        of the document) is never touched.
        """
        lines = raw_text.split("\n")
        if len(lines) >= 2:
            first, second = lines[0].strip(), lines[1].strip()
            if cls._RUNNING_HEADER_RE.match(first) and cls._PAGE_NUMBER_LINE_RE.match(second):
                return "\n".join(lines[2:])
            if cls._PAGE_NUMBER_LINE_RE.match(first) and cls._RUNNING_HEADER_RE.match(second):
                return "\n".join(lines[2:])
        return raw_text

    @classmethod
    def _detect_entry_header(cls, raw_text: str) -> Optional[Dict[str, str]]:
        """
        If this page's raw (line-preserved) text contains a disease-entry
        boundary marker, return {condition_name, category, priority,
        icd10_code}; otherwise None. Operates on RAW text (real newlines),
        not the whitespace-collapsed `text` used for chunking content,
        because the title/Category-line adjacency this relies on only
        survives before normalize_text() collapses all whitespace to single
        spaces.
        """
        lines = raw_text.split("\n")
        for i, line in enumerate(lines):
            match = cls._ENTRY_HEADER_RE.match(line.strip())
            if not match:
                continue
            title = None
            j = i - 1
            while j >= 0:
                candidate = lines[j].strip()
                if candidate:
                    title = candidate
                    break
                j -= 1
            if title:
                return {
                    "condition_name": title,
                    "category": match.group("category").strip(),
                    "priority": match.group("priority").strip(),
                    "icd10_code": match.group("icd10").strip(),
                }
        return None

    def extract_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        reader = pypdf.PdfReader(str(pdf_path))
        pages: List[Dict[str, Any]] = []
        for index, page in enumerate(reader.pages, start=1):
            raw_text = self._strip_running_header(page.extract_text() or "")
            text = normalize_text(raw_text)
            if not text:
                continue
            pages.append(
                {
                    "page": index,
                    "text": text,
                    "raw_text": raw_text,
                }
            )
        return pages

    def chunk_pdf(
        self,
        pdf_path: Path,
        document_name: str = DEFAULT_DOCUMENT_NAME
    ) -> List[PDFTextChunk]:
        pages = self.extract_pages(pdf_path)
        chunks: List[PDFTextChunk] = []
        chunk_index = 0

        # Tracks which disease entry we're currently inside as we walk the
        # document in page order — updated whenever a page contains a new
        # entry-boundary marker, and carried forward onto every subsequent
        # chunk (across page boundaries) until the next one is found. This
        # is what lets retrieval later fetch "every chunk belonging to
        # Atrial Fibrillation" as a direct lookup instead of hoping semantic
        # ranking happens to surface all of them.
        current_condition: Optional[Dict[str, str]] = None
        condition_chunk_order = 0

        for page in pages:
            page_number = int(page["page"])

            entry_header = self._detect_entry_header(page.get("raw_text", ""))
            if entry_header is not None:
                current_condition = entry_header
                condition_chunk_order = 0

            page_words = words(page["text"])
            if len(page_words) < 20:
                continue

            cursor = 0
            while cursor < len(page_words):
                chunk_words = page_words[cursor: cursor + self.max_words]
                content = " ".join(chunk_words).strip()
                if len(content) >= 120:
                    metadata: Dict[str, Any] = {
                        "source_type": "pdf",
                        "embedding_model": EMBEDDING_MODEL_NAME,
                        "chunking": {
                            "strategy": "page_bounded_sliding_window",
                            "max_words": self.max_words,
                            "overlap_words": self.overlap_words,
                        },
                    }
                    if current_condition is not None:
                        metadata["condition_name"] = current_condition["condition_name"]
                        metadata["category"] = current_condition["category"]
                        metadata["priority"] = current_condition["priority"]
                        metadata["icd10_code"] = current_condition["icd10_code"]
                        metadata["condition_chunk_order"] = condition_chunk_order
                        condition_chunk_order += 1

                    chunks.append(
                        PDFTextChunk(
                            document_name=document_name,
                            chunk_id=stable_chunk_id(
                                document_name,
                                page_number,
                                page_number,
                                chunk_index,
                                content,
                            ),
                            page_start=page_number,
                            page_end=page_number,
                            chunk_index=chunk_index,
                            section_hint=infer_section_hint(content),
                            content=content,
                            metadata_json=metadata,
                        )
                    )
                    chunk_index += 1

                if cursor + self.max_words >= len(page_words):
                    break
                cursor += max(1, self.max_words - self.overlap_words)

        return chunks


class PDFKnowledgeIngestor:
    def __init__(self, db: Optional[ECGDatabase] = None):
        self.db = db or ECGDatabase()
        self.chunker = PDFSemanticChunker()
        self.model = get_embedding_model()

    def ingest(
        self,
        pdf_path: Path = DEFAULT_PDF_PATH,
        document_name: str = DEFAULT_DOCUMENT_NAME,
        replace_document: bool = True,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        chunks = self.chunker.chunk_pdf(pdf_path, document_name=document_name)
        if not chunks:
            return {
                "document_name": document_name,
                "chunk_count": 0,
                "inserted": 0,
            }

        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        insert_rows = [
            chunk.to_insert_dict(embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        inserted = self.db.insert_pdf_knowledge_chunks(
            insert_rows,
            replace_document=replace_document,
        )
        return {
            "document_name": document_name,
            "chunk_count": len(chunks),
            "inserted": inserted,
            "pages": {
                "start": min(chunk.page_start for chunk in chunks),
                "end": max(chunk.page_end for chunk in chunks),
            },
        }


class PDFSemanticRetriever:
    def __init__(
        self,
        db: Optional[ECGDatabase] = None,
        document_name: str = DEFAULT_DOCUMENT_NAME
    ):
        self.db = db or ECGDatabase()
        self.document_name = document_name
        self.model = None

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        if self.model is None:
            self.model = get_embedding_model()
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        return [
            dict(row)
            for row in self.db.search_pdf_knowledge(
                embedding,
                top_k=top_k,
                document_name=self.document_name,
            )
        ]

    # DEFAULT_FULL_ENTRY_SCAN: measured directly against this KB's actual
    # chunker output (chunking is per-page — most pages produce exactly one
    # chunk each — so entry chunk-count tracks its page-span closely): 57
    # entries, 3-6 pages each (avg 4.4), producing 3-6 chunks each (avg 4.3,
    # max 6 for the longest entry). 30 gives 5x headroom over the largest
    # entry seen, while a query built from the disease's own name should
    # rank that disease's own chunks far above unrelated ones anyway.
    # Requires no new SQL or schema — reuses search_pdf_knowledge exactly as
    # retrieve() above does, just with a larger top_k, then filters and
    # reorders the results in Python.
    DEFAULT_FULL_ENTRY_SCAN = 30

    def retrieve_full_condition(
        self,
        condition_name: str,
        max_scan: int = DEFAULT_FULL_ENTRY_SCAN,
    ) -> List[Dict[str, Any]]:
        """
        Fetch every ingested chunk belonging to ONE specific disease entry —
        its complete write-up (Definition through References), not just the
        top-k most semantically similar chunks. This exists because top-k
        similarity ranking can exclude a real chunk of the right disease
        (e.g. its "Treatment Overview" chunk scoring lower than an unrelated
        disease's more query-similar chunk), especially once the same top-k
        budget is being split across multiple candidate diseases in one
        event. Fetching by tagged identity instead of by ranking avoids that
        failure mode entirely for the one disease that matters most.

        Returns [] (not an exception) if:
          - the KB hasn't been re-ingested since condition-name tagging was
            added to PDFSemanticChunker (existing rows in the DB predate the
            tag and won't have metadata_json['condition_name'] set), or
          - condition_name doesn't match any tagged entry in this document
            (e.g. a generic event with no specific book-listed disease).
        Callers should fall back to retrieve() in either case — see
        PDFSemanticECGAgent.analyze().
        """
        if not condition_name:
            return []
        if self.model is None:
            self.model = get_embedding_model()
        embedding = self.model.encode(condition_name, normalize_embeddings=True)
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        candidates = [
            dict(row)
            for row in self.db.search_pdf_knowledge(
                embedding,
                top_k=max_scan,
                document_name=self.document_name,
            )
        ]

        target = _normalize_disease_name(condition_name)
        matched: List[Dict[str, Any]] = []
        for row in candidates:
            metadata = row.get("metadata_json") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}
            tagged_name = metadata.get("condition_name")
            if not tagged_name:
                continue
            if _normalize_disease_name(tagged_name) == target:
                row["condition_name"] = tagged_name
                row["condition_chunk_order"] = metadata.get("condition_chunk_order", 0)
                row["category"] = metadata.get("category")
                row["priority"] = metadata.get("priority")
                row["icd10_code"] = metadata.get("icd10_code")
                matched.append(row)

        # Exact equality only (not the looser containment matching
        # _disease_names_match uses for validation) — here we're picking
        # ONE specific tagged entry out of 57, where a looser match risks
        # pulling in a different-but-similarly-named condition's pages.
        # Validation's looser matching is fine because it only ever checks
        # "is this name plausibly one of a small handful of already-known
        # candidates", a different and lower-stakes question.
        matched.sort(key=lambda r: (r.get("page_start") or 0, r.get("condition_chunk_order") or 0))
        return matched


class PDFSemanticPromptBuilder:
    # NOTE: this prompt intentionally does NOT ask for or expect raw beat-level
    # waveform data, rolling-window statistics, or a signal-quality score. The
    # upstream deterministic pipeline (Event_Manager.py + disease_detector.py +
    # temporal_analysis.py) has already evaluated those and distilled the
    # result into the event record's own fields (confidence, severity, reason,
    # ecg_findings, missing_data). That distillation IS the ECG evidence this
    # prompt reasons over — sending the raw beat/window/signal-quality data on
    # top of it would be redundant, not additive.
    SYSTEM_PROMPT = """
You are a clinical ECG decision-support reasoning component.

You reason only from three sources, all supplied below:
1. EVENT RECORD — the deterministic pipeline's own evidence for the event
   that triggered you (rule-engine / disease-detector confidence, severity,
   reason, specific ECG findings already identified, and any data the
   detector itself flagged as missing). Treat this as your ECG evidence.
   You do not receive raw beats, rolling-window statistics, or a signal
   quality score — those were already used upstream to produce this record.
2. PATIENT METADATA AND CONFIRMED SYMPTOMS, plus RECENT EVENT HISTORY for
   temporal context.
3. RETRIEVED KNOWLEDGE — excerpts from Cardiac_Diagnostics_Comprehensive_KB.pdf,
   a 57-condition reference where every disease entry follows the same
   12-section template: Definition, Mechanism, ECG Findings, Required ECG
   Features, Diagnostic Criteria, Differential Diagnosis, Symptoms, Risk
   Factors, Immediate Actions, Treatment Overview, Follow-up Questions,
   References. Each retrieved chunk is tagged with the section it came from
   — use that tag to know what kind of claim it can support (e.g. only an
   "Immediate Actions" or "Treatment Overview" chunk grounds a recommended
   action; only a "Differential Diagnosis" chunk grounds a differential).

REASONING STEPS

1. Evidence completeness — look at the event record's own missing_data (or
   equivalent) and the patient's unanswered symptom questions. Note what is
   incomplete before concluding anything. This is your data-quality gate;
   you do not have a separate signal-quality score to check.
2. Validate the event — compare the event record's reason and ECG findings
   against retrieved "ECG Findings" / "Required ECG Features" / "Diagnostic
   Criteria" chunks. Classify as supported, uncertain, rejected, or
   emergency.
3. Disease matching — weigh the event's primary candidate against any other
   candidates present in the event record (do not introduce a condition
   that is not already present in the event record).
4. Differential diagnosis — use retrieved "Differential Diagnosis" chunks to
   name real alternatives and why they rank lower, not just the top pick.
5. Risk assessment — combine retrieved "Risk Factors" content with the
   patient's comorbidities/age/sex.
6. Immediate action — ground any recommendation in retrieved "Immediate
   Actions" / "Treatment Overview" chunks. Do not recommend anything you
   cannot point to a chunk for.
7. Follow-up questions — pull from the event record's own symptom questions
   and retrieved "Follow-up Questions" chunks, but only include ones that
   are not already answered in the patient's confirmed symptoms and that
   would actually change the assessment.
8. Confidence calibration — your event_confidence and per-condition
   confidence may only be equal to or lower than the event record's own
   rule-engine confidence, never higher, and must be lowered further when
   step 1 found incomplete evidence.

STRICT RULES

Never claim a definitive diagnosis.
Never invent ECG findings, symptoms, medications, patient history, or a
condition that is not already present in the event record.
Every entry in evidence.supporting / evidence.contradicting must cite a
specific source: either a field from the event record or patient metadata,
or a retrieved chunk_id.
If something cannot be determined from the supplied evidence, write exactly
"Cannot determine from provided evidence." in the relevant field instead of
guessing or omitting it.

OUTPUT CONTRACT

Respond with exactly one JSON object and nothing else — no markdown, no
prose before or after it, no code fences. It must match this shape:

{
  "event_assessment": "supported | uncertain | rejected | emergency",
  "event_confidence": 0-100,
  "summary": "1-3 sentence plain-language summary",
  "most_likely_conditions": [
    {"condition": "string, must already appear in the event record",
     "subtype": "string, or the exact phrase: Cannot determine from provided evidence.",
     "confidence": 0-100,
     "icd10_codes": ["string", "..."]}
  ],
  "evidence": {
    "supporting": [{"finding": "string", "source": "event field name, or chunk_id"}],
    "contradicting": [{"finding": "string", "source": "event field name, or chunk_id"}],
    "missing": ["string"]
  },
  "differential_diagnosis": [
    {"condition": "string", "why_considered": "string", "why_ranked_lower": "string"}
  ],
  "risk_assessment": {
    "stroke": "low | moderate | high | critical | not_applicable",
    "sudden_cardiac_death": "low | moderate | high | critical | not_applicable",
    "heart_failure": "low | moderate | high | critical | not_applicable",
    "cardiogenic_shock": "low | moderate | high | critical | not_applicable",
    "overall_urgency": "routine | monitor | urgent | emergency"
  },
  "recommended_questions": [
    {"question": "string", "why_it_matters": "string"}
  ],
  "immediate_recommendations": ["string"],
  "backend_flags": {"notify_clinician_now": true or false},
  "follow_up": {"recheck_in_minutes": 0, "recheck_condition": "string or null"},
  "references": [{"citation_index": 1, "chunk_id": "string"}],
  "warnings": ["string"],
  "reasoning_summary": "short audit trail of steps 1-8 above, not a raw chain of thought"
}

Output nothing outside that single JSON object.
"""

    @staticmethod
    def _safe_json(value: Any) -> str:
        try:
            return json.dumps(value, indent=2, default=str, ensure_ascii=False)
        except Exception:
            return str(value)

    @staticmethod
    def format_chunks(chunks: List[Dict[str, Any]]) -> str:
        lines = []
        for index, chunk in enumerate(chunks, start=1):
            content = chunk.get("content") or ""
            lines.append(
                "\n".join(
                    [
                        f"[{index}] chunk_id={chunk.get('chunk_id')}",
                        f"document={chunk.get('document_name')} pages={chunk.get('page_start')}-{chunk.get('page_end')}",
                        f"section_hint={chunk.get('section_hint')} similarity={float(chunk.get('similarity') or 0):.3f}",
                        content,
                    ]
                )
            )
        return "\n\n".join(lines)

    @classmethod
    def build(
        cls,
        session_id: str,
        event_type: str,
        query: str,
        event: Optional[Dict[str, Any]],
        patient_metadata: Optional[Dict[str, Any]],
        recent_events: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        user_prompt = f"""
ECG EVENT
Session: {session_id}
Event type: {event_type}

EVENT RECORD (this is your ECG evidence — confidence, reason, findings, and
missing_data below were already produced by the deterministic rule engine /
disease detector; no raw beat or window data is supplied on top of it)
{cls._safe_json(event or {})}

PATIENT METADATA AND CONFIRMED SYMPTOMS
{cls._safe_json(patient_metadata or {})}

RECENT EVENT HISTORY (temporal context, most recent last)
{cls._safe_json(recent_events or [])}

RETRIEVED KNOWLEDGE (Cardiac_Diagnostics_Comprehensive_KB.pdf)
{cls.format_chunks(chunks)}

TASK
Follow the 8 reasoning steps and strict rules from the system prompt, then
respond with exactly one JSON object matching the schema given there. No
markdown, no prose outside the JSON object.
"""
        return {
            "system": cls.SYSTEM_PROMPT,
            "user": user_prompt,
        }


_REQUIRED_ASSESSMENT_KEYS = (
    "event_assessment",
    "event_confidence",
    "summary",
    "most_likely_conditions",
    "evidence",
    "differential_diagnosis",
    "risk_assessment",
    "recommended_questions",
    "immediate_recommendations",
    "backend_flags",
    "follow_up",
    "references",
    "warnings",
    "reasoning_summary",
)


def _strip_code_fence(text: str) -> str:
    """Defensive cleanup for models that wrap JSON in a ```json fence anyway."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"```\s*$", "", stripped)
    return stripped.strip()


def _parse_json_assessment(llm_text: Optional[str]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse the model's raw text as a single JSON object.

    Returns (assessment_dict, error_message) and never raises — a malformed
    or non-JSON response is an expected possible model failure mode, not a
    bug in this code, so it must degrade to (None, "...") rather than throw.
    """
    if not llm_text:
        return None, "empty response from LLM"

    candidate = _strip_code_fence(llm_text)

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Some models add a stray sentence before/after the JSON object even
        # when told not to. Fall back to extracting the outermost {...}.
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return None, "no JSON object found in response"
        try:
            parsed = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return None, f"invalid JSON: {exc}"

    if not isinstance(parsed, dict):
        return None, "response was valid JSON but not a JSON object"

    return parsed, None


def _normalize_disease_name(name: Optional[str]) -> str:
    """
    Normalize a disease name for comparison, not for display.

    The event record and the model are allowed to name the same condition
    differently and both be correct — e.g. the event record's
    "Ventricular Tachycardia (VT) — Nonsustained" vs. the model's
    "ventricular_tachycardia" (with "Nonsustained" correctly placed in the
    separate `subtype` field instead, which is the schema working as
    intended, not a hallucination). This strips exactly the kind of
    formatting difference that caused that false positive:
      - case
      - parenthetical abbreviations, e.g. "(VT)"
      - a trailing " — subtype" qualifier after an em-dash
      - underscores used instead of spaces
    Hyphens inside a name (e.g. "Wolff-Parkinson-White") are deliberately
    left alone so real compound disease names are not broken apart.
    """
    if not name:
        return ""
    text = str(name).strip().lower()
    text = re.sub(r"\([^)]*\)", " ", text)  # drop parenthetical abbreviations
    text = text.split("—", 1)[0]  # drop a trailing " — subtype" qualifier
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)  # strip stray punctuation, keep hyphens
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _disease_names_match(name_a: str, name_b: str) -> bool:
    """True if two normalized disease names are the same condition, allowing
    one to be a superset of the other (e.g. extra qualifier words)."""
    if not name_a or not name_b:
        return False
    return name_a == name_b or name_a in name_b or name_b in name_a


def _candidate_diseases_from_event(event: Optional[Dict[str, Any]]) -> set[str]:
    """
    All disease names already present in the event record, used to check the
    model didn't invent a condition. Covers both a single primary disease
    (metadata_json['disease']) and any secondary detector results merged in
    by temporal_analysis._merge_disease_events
    (metadata_json['disease_detector_results']).

    Returned names are normalized (see _normalize_disease_name) — this set
    is for comparison only, never for display.
    """
    event = event or {}
    metadata = event.get("metadata_json") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    names: set[str] = set()
    primary = metadata.get("disease") or metadata.get("condition")
    if primary:
        names.add(_normalize_disease_name(primary))

    for extra in metadata.get("disease_detector_results") or []:
        if isinstance(extra, dict):
            extra_disease = extra.get("disease")
            if extra_disease:
                names.add(_normalize_disease_name(extra_disease))

    names.discard("")
    return names


def _validate_assessment(
    assessment: Dict[str, Any],
    event: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Cheap, deterministic post-generation checks — catches the two failure
    modes most likely from a small instruct model under a strict schema:
    inventing a condition that isn't in the event record, and setting the
    clinician-notification flag inconsistently with its own stated urgency.
    Never raises; always returns a report, so a malformed model response
    degrades gracefully instead of crashing the pipeline.
    """
    issues: List[str] = []

    missing_keys = [key for key in _REQUIRED_ASSESSMENT_KEYS if key not in assessment]
    if missing_keys:
        issues.append(f"missing expected keys: {missing_keys}")

    known_diseases = _candidate_diseases_from_event(event)
    conditions = assessment.get("most_likely_conditions")
    if known_diseases and isinstance(conditions, list):
        for entry in conditions:
            if not isinstance(entry, dict):
                continue
            named = _normalize_disease_name(entry.get("condition", ""))
            if named and not any(_disease_names_match(named, known) for known in known_diseases):
                issues.append(
                    f"most_likely_conditions names '{entry.get('condition')}', "
                    f"which is not present in the event record ({sorted(known_diseases)})"
                )

    risk = assessment.get("risk_assessment")
    flags = assessment.get("backend_flags")
    if isinstance(risk, dict) and isinstance(flags, dict):
        urgency = str(risk.get("overall_urgency", "")).lower()
        notify = bool(flags.get("notify_clinician_now"))
        if notify and urgency not in ("urgent", "emergency"):
            issues.append(
                "backend_flags.notify_clinician_now is true but "
                f"risk_assessment.overall_urgency is '{urgency}'"
            )

    return {"valid": not issues, "issues": issues}


class PDFSemanticECGAgent:
    """
    Pipeline-compatible agent that uses only semantic pgvector retrieval over the PDF.
    """

    def __init__(
        self,
        db: Optional[ECGDatabase] = None,
        llm: Optional[Any] = None,
        llm_mode: str = "inference",
        local_model_path: Optional[str] = None,
        document_name: str = DEFAULT_DOCUMENT_NAME,
        top_k: int = DEFAULT_TOP_K,
        formatter_llm: Optional[Any] = None,
    ):
        self.db = db or ECGDatabase()
        self.llm = llm or build_llm_client(
            mode=llm_mode,
            local_model_path=local_model_path,
        )
        # The reasoning call above (self.llm, used in analyze() below) and
        # the report-rendering call are deliberately separable. By default
        # this reuses the same client/model — one fewer thing to configure —
        # but you can pass a genuinely different LLMClient here (e.g. a
        # different, cheaper, or faster model) since rendering an
        # already-decided JSON assessment into prose is a much narrower task
        # than the clinical reasoning call and doesn't need the same model.
        self.formatter_llm = formatter_llm or self.llm
        self.document_name = document_name
        self.top_k = top_k
        self.retriever = PDFSemanticRetriever(
            db=self.db,
            document_name=document_name,
        )

    @staticmethod
    def _extract_event_metadata(event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse metadata_json off an event record, tolerating it arriving
        as a dict or as a raw JSON string (see database.py's JSONB columns)."""
        event = event or {}
        metadata = event.get("metadata_json") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {"raw_metadata": metadata}
        return metadata

    @classmethod
    def _extract_primary_disease(cls, event: Optional[Dict[str, Any]]) -> str:
        """The primary disease name attached to this event, if any — used
        both to build the retrieval query and, separately in analyze(), to
        decide whether a full-entry fetch (retrieve_full_condition) is even
        possible for this event."""
        metadata = cls._extract_event_metadata(event)
        return str(metadata.get("disease") or metadata.get("condition") or "").strip()

    @staticmethod
    def build_query(
        event_type: str,
        event: Optional[Dict[str, Any]],
        patient_metadata: Optional[Dict[str, Any]],
        recent_events: List[Dict[str, Any]],
    ) -> str:
        # NOTE: no beat/window/signal-quality data goes into the retrieval
        # query either — the event's own disease/findings/missing_data
        # already say what this event is about, which is what retrieval
        # needs to find the right knowledge-base sections.
        metadata = PDFSemanticECGAgent._extract_event_metadata(event)

        disease = metadata.get("disease") or metadata.get("condition") or ""
        confirmation = metadata.get("confirmation") or {}
        findings = metadata.get("ecg_findings") or []
        missing = metadata.get("missing_data") or []

        query_parts = [
            f"event type {event_type}",
            f"possible disease {disease}" if disease else "",
            "ecg findings " + " ".join(map(str, findings)) if findings else "",
            "missing evidence " + " ".join(map(str, missing)) if missing else "",
            "confirmation " + json.dumps(confirmation, default=str) if confirmation else "",
            "patient " + json.dumps(patient_metadata or {}, default=str),
        ]
        if recent_events:
            query_parts.append(
                "recent events " + " ".join(
                    str(item.get("event_type"))
                    for item in recent_events
                    if item.get("event_type")
                )
            )
        return "\n".join(part for part in query_parts if part)

    def analyze(
        self,
        session_id: str,
        event_type: str,
        event: Optional[Dict[str, Any]] = None,
        beat_data: Optional[Dict[str, Any]] = None,
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # `beat_data` is accepted only so existing callers (ecg_pipeline.py
        # currently passes beat_data=beat_data) keep working unchanged. It is
        # intentionally not read: the triggering event's own reason /
        # ecg_findings / confidence (already produced upstream by
        # disease_detector.py / Event_Manager.py) is the ECG evidence this
        # agent reasons over, so the single triggering beat, rolling-window
        # statistics, and signal-quality score are no longer fetched or sent.
        del beat_data

        recent_events = self.db.get_recent_events(session_id, limit=10000)
        if patient_metadata is None:
            patient_metadata = self.db.get_patient_metadata(session_id)

        query = self.build_query(
            event_type=event_type,
            event=event,
            patient_metadata=patient_metadata,
            recent_events=recent_events,
        )

        # Try to fetch the COMPLETE book entry for the event's primary
        # disease first (all ~3-6 pages of it — Definition through
        # References), rather than only the top-k semantically-ranked
        # chunks, which can exclude a real chunk of the right disease (e.g.
        # its own "Treatment Overview" section scoring lower than an
        # unrelated disease's more query-similar text). See
        # PDFSemanticRetriever.retrieve_full_condition() for how this is
        # matched. Supplement with the normal semantic search too, so
        # content relevant to a *secondary* candidate disease (differential
        # diagnosis material, etc.) isn't lost — merged in only if it isn't
        # already covered by the full entry.
        primary_disease = self._extract_primary_disease(event)
        full_entry_chunks: List[Dict[str, Any]] = []
        if primary_disease:
            full_entry_chunks = self.retriever.retrieve_full_condition(primary_disease)

        semantic_chunks = self.retriever.retrieve(query, top_k=self.top_k)

        if full_entry_chunks:
            seen_ids = {c.get("chunk_id") for c in full_entry_chunks}
            supplemental = [c for c in semantic_chunks if c.get("chunk_id") not in seen_ids]
            chunks = full_entry_chunks + supplemental
            retrieval_mode = "full_entry"
        else:
            # Either there's no primary disease name on this event (e.g. a
            # generic/rate-based event), or the KB hasn't been re-ingested
            # since condition-name tagging was added — either way, fall back
            # to exactly the retrieval behavior this agent already had.
            chunks = semantic_chunks
            retrieval_mode = "semantic_top_k"

        if not chunks:
            response = {
                "event_assessment": "uncertain",
                "reason": "No semantic PDF knowledge chunks were retrieved. Run ingest_pdf_knowledge.py first.",
                "query": query,
                "retrieval_mode": retrieval_mode,
                "primary_disease": primary_disease or None,
                "retrieved_chunks": [],
                "assessment": None,
                "validation": {"valid": False, "issues": ["no retrieved chunks — assessment not generated"]},
            }
            # No assessment exists in this branch at all, so there is
            # nothing for the render-LLM to work from — go straight to the
            # deterministic fallback rather than making a pointless LLM call,
            # for both the doctor and patient reports.
            response["formatted_report_doctor"] = render_report_or_fallback_en(
                response, event=event, document_name=self.document_name
            )
            response["formatted_report_doctor_source"] = "template_fallback"
            response["formatted_report_patient"] = (
                "Your care team is reviewing this result and will follow up with you."
            )
            response["formatted_report_patient_source"] = "template_fallback"
            self.db.insert_agent_interaction(
                patient_id=session_id,
                session_id=session_id,
                event_type=event_type,
                prompt_json={"query": query, "rag_status": "empty"},
                retrieved_chunks=[],
                response_json=response,
            )
            return response

        prompt = PDFSemanticPromptBuilder.build(
            session_id=session_id,
            event_type=event_type,
            query=query,
            event=event,
            patient_metadata=patient_metadata,
            recent_events=recent_events,
            chunks=chunks,
        )
        llm_text = self.llm.generate(
            prompt["system"],
            prompt["user"],
        )

        assessment, parse_error = _parse_json_assessment(llm_text)
        if assessment is None:
            # One bounded retry with a stricter reminder — small instruct
            # models occasionally wrap JSON in prose or a code fence on the
            # first attempt despite the instruction not to.
            retry_user_prompt = (
                prompt["user"]
                + "\n\nYour previous response was not a single valid JSON "
                "object. Respond again with ONLY the JSON object described "
                "above — no markdown, no code fence, no other text."
            )
            llm_text_retry = self.llm.generate(prompt["system"], retry_user_prompt)
            assessment_retry, parse_error_retry = _parse_json_assessment(llm_text_retry)
            if assessment_retry is not None:
                llm_text = llm_text_retry
                assessment = assessment_retry
                parse_error = None
            else:
                parse_error = parse_error_retry or parse_error

        retrieved_chunks = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "section_hint": chunk.get("section_hint"),
                "similarity": float(chunk.get("similarity") or 0.0),
                "condition_name": chunk.get("condition_name"),
            }
            for chunk in chunks
        ]

        response: Dict[str, Any] = {
            "rag_type": "pdf_semantic_pgvector",
            "document_name": self.document_name,
            "query": query,
            "retrieval_mode": retrieval_mode,
            "primary_disease": primary_disease or None,
            "retrieved_chunks": retrieved_chunks,
            "llm_response": llm_text,
            "assessment": assessment,
        }
        if assessment is not None:
            response["validation"] = _validate_assessment(assessment, event)
        else:
            response["validation"] = {
                "valid": False,
                "issues": [f"model did not return parseable JSON: {parse_error}"],
            }

        # Two separate human-readable reports are generated from the SAME
        # `assessment` JSON — one for clinical staff, one for the patient.
        # They are NOT the same text with different formatting: the
        # patient version deliberately omits differential diagnoses,
        # confidence percentages, ICD codes, lab/medication specifics, and
        # citations (see llm_report_renderer.py's PATIENT_REPORT_SYSTEM_PROMPT
        # and its content-safety scan for why those are actively screened
        # out, not just "left out by convention"). Each has its own
        # LLM-render-then-deterministic-fallback pair, independent of the
        # other — a failure generating one must never block or corrupt the
        # other.
        #
        # NOTE: this replaces the earlier `formatted_report` /
        # `formatted_report_source` field names with `formatted_report_doctor`
        # / `formatted_report_doctor_source`, to sit symmetrically alongside
        # the new `formatted_report_patient` / `formatted_report_patient_source`
        # — update anything already reading the old field names.
        doctor_report_text: Optional[str] = None
        doctor_render_error: Optional[str] = None
        if assessment is not None:
            try:
                doctor_report_text, doctor_render_error = render_via_llm(
                    self.formatter_llm,
                    assessment,
                    event=event,
                    document_name=self.document_name,
                )
            except Exception as exc:
                # The formatter LLM call itself failing outright (network,
                # auth, etc.) must not take down the whole agent response —
                # fall back to the deterministic template exactly as if it
                # had just returned a malformed report.
                doctor_render_error = f"formatter LLM call raised: {exc}"

        if doctor_report_text is not None:
            response["formatted_report_doctor"] = doctor_report_text
            response["formatted_report_doctor_source"] = "llm"
        else:
            response["formatted_report_doctor"] = render_report_or_fallback_en(
                response, event=event, document_name=self.document_name
            )
            response["formatted_report_doctor_source"] = "template_fallback"
            if doctor_render_error:
                response["formatted_report_doctor_render_error"] = doctor_render_error

        patient_report_text: Optional[str] = None
        patient_render_error: Optional[str] = None
        if assessment is not None:
            try:
                patient_report_text, patient_render_error = render_patient_report_via_llm(
                    self.formatter_llm,
                    assessment,
                    event=event,
                )
            except Exception as exc:
                patient_render_error = f"patient formatter LLM call raised: {exc}"

        if patient_report_text is not None:
            response["formatted_report_patient"] = patient_report_text
            response["formatted_report_patient_source"] = "llm"
        else:
            # render_patient_summary_en() is guaranteed free of the content
            # the patient prompt is forbidden from including, by
            # construction rather than by prompting — the same safety
            # guarantee the doctor-report fallback has, just for the
            # patient-safety rules instead of the hallucination rules.
            response["formatted_report_patient"] = render_patient_summary_en(
                assessment, event=event
            ) if assessment is not None else (
                "Your care team is reviewing this result and will follow up with you."
            )
            response["formatted_report_patient_source"] = "template_fallback"
            if patient_render_error:
                response["formatted_report_patient_render_error"] = patient_render_error

        self.db.insert_agent_interaction(
            patient_id=session_id,
            session_id=session_id,
            event_type=event_type,
            prompt_json=prompt,
            retrieved_chunks=retrieved_chunks,
            response_json=response,
        )
        return response


def ingest_pdf_knowledge(
    pdf_path: Path = DEFAULT_PDF_PATH,
    document_name: str = DEFAULT_DOCUMENT_NAME,
    replace_document: bool = True,
) -> Dict[str, Any]:
    ingestor = PDFKnowledgeIngestor()
    return ingestor.ingest(
        pdf_path=pdf_path,
        document_name=document_name,
        replace_document=replace_document,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic PDF RAG ingestion and search.")
    parser.add_argument("--ingest", action="store_true", help="Extract, embed, and store PDF chunks in PostgreSQL.")
    parser.add_argument("--search", type=str, help="Semantic search query.")
    parser.add_argument("--analyze-event", type=str, help="Run full RAG agent analysis for this event type.")
    parser.add_argument("--session-id", default="manual_semantic_rag_test", help="Session id for --analyze-event.")
    parser.add_argument("--event-json", type=str, help="Optional JSON object with event details for --analyze-event.")
    parser.add_argument("--beat-json", type=str, help="Optional JSON object with latest beat measurements for --analyze-event.")
    parser.add_argument("--patient-json", type=str, help="Optional JSON object with patient metadata for --analyze-event.")
    parser.add_argument("--test", action="store_true", help="Use local Transformers LLM backend for generation.")
    parser.add_argument("--inference", action="store_true", help="Use Hugging Face Inference API backend for generation.")
    parser.add_argument("--local-model-path", help="Local model folder for --test, e.g. Phi_3_5_mini_instruct.")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF_PATH), help="PDF path.")
    parser.add_argument("--document-name", default=DEFAULT_DOCUMENT_NAME, help="Document name stored in DB.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--no-replace", action="store_true", help="Do not delete existing chunks for the document before insert.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.ingest:
        result = ingest_pdf_knowledge(
            pdf_path=Path(args.pdf),
            document_name=args.document_name,
            replace_document=not args.no_replace,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.search:
        retriever = PDFSemanticRetriever(document_name=args.document_name)
        rows = retriever.retrieve(args.search, top_k=args.top_k)
        for index, row in enumerate(rows, start=1):
            print(
                f"[{index}] sim={float(row.get('similarity') or 0):.3f} "
                f"pages={row.get('page_start')}-{row.get('page_end')} "
                f"section={row.get('section_hint')} "
                f"id={row.get('chunk_id')}"
            )
            print((row.get("content") or "")[:500].replace("\n", " "))
            print()
        return 0

    if args.analyze_event:
        if args.test and args.inference:
            raise ValueError("Use only one LLM mode: --test or --inference.")

        llm_mode = "test" if args.test else "inference"

        def parse_json_arg(raw: Optional[str], fallback: Dict[str, Any]) -> Dict[str, Any]:
            if not raw:
                return fallback
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("JSON CLI arguments must be objects.")
            return parsed

        event = parse_json_arg(
            args.event_json,
            {
                "event_type": args.analyze_event,
                "metadata_json": {
                    "disease": args.analyze_event.replace("_", " "),
                },
            },
        )
        beat_data = parse_json_arg(args.beat_json, {})
        patient_metadata = parse_json_arg(args.patient_json, {})

        agent = PDFSemanticECGAgent(
            llm_mode=llm_mode,
            local_model_path=args.local_model_path,
            document_name=args.document_name,
            top_k=args.top_k,
        )
        response = agent.analyze(
            session_id=args.session_id,
            event_type=args.analyze_event,
            event=event,
            beat_data=beat_data,
            patient_metadata=patient_metadata,
        )
        print(json.dumps(response, indent=2, default=str))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())