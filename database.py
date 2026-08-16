import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager

from config import DB_CONFIG as _DB_CONFIG

logger = logging.getLogger(__name__)

class ECGDatabase:
    def __init__(self, db_config: Dict[str, str] = None, pool_min: int = 1, pool_max: int = 10):
        """
        Initialize PostgreSQL database connection.
        
        Args:
            db_config: Dictionary with database connection parameters.
                Falls back to DB_CONFIG from config.py (loaded from .env).
            pool_min: Minimum number of connections in pool
            pool_max: Maximum number of connections in pool
        """
        self.db_config = dict(_DB_CONFIG)
        if db_config:
            self.db_config.update(
                (k, v) for k, v in db_config.items() if v is not None
            )
        
        # Create connection pool
        self.pool = SimpleConnectionPool(
            pool_min, 
            pool_max,
            **self.db_config
        )
        
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections from pool."""
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            self.pool.putconn(conn)

    def _init_db(self):
        """Initializes the database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Enable pgvector
            cursor.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
            """)
            
            # Table for beat-level features
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS beat_features (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp DOUBLE PRECISION NOT NULL,
                    beat_index INTEGER NOT NULL,
                    predicted_label INTEGER,
                    prediction_confidence DOUBLE PRECISION,
                    rr_interval DOUBLE PRECISION,
                    heart_rate DOUBLE PRECISION,
                    qrs_width DOUBLE PRECISION,
                    qrs_voltage DOUBLE PRECISION,
                    q_amplitude DOUBLE PRECISION,
                    r_amplitude DOUBLE PRECISION,
                    s_amplitude DOUBLE PRECISION,
                    q_peak INTEGER,
                    s_peak INTEGER,
                    qt_interval DOUBLE PRECISION,
                    qtc DOUBLE PRECISION,
                    qtc_bazett DOUBLE PRECISION,
                    qtc_fridericia DOUBLE PRECISION,
                    st_deviation DOUBLE PRECISION,
                    st_segment_ms DOUBLE PRECISION,
                    t_wave_min DOUBLE PRECISION,
                    t_wave_inverted BOOLEAN,
                    t_wave_width_ms DOUBLE PRECISION,
                    t_wave_amplitude DOUBLE PRECISION,
                    t_wave_polarity TEXT,
                    tpeak_tend_interval_ms DOUBLE PRECISION,
                    p_wave_detected BOOLEAN,
                    p_wave_width_ms DOUBLE PRECISION,
                    p_wave_prominence DOUBLE PRECISION,
                    p_wave_inverted BOOLEAN,
                    p_wave_amplitude DOUBLE PRECISION,
                    p_wave_polarity TEXT,
                    pr_interval_ms DOUBLE PRECISION,
                    pr_segment_ms DOUBLE PRECISION,
                    u_wave_detected BOOLEAN,
                    u_wave_peak INTEGER,
                    u_wave_amplitude DOUBLE PRECISION,
                    delta_wave_detected BOOLEAN,
                    delta_wave_slope DOUBLE PRECISION,
                    flutter_baseline_power DOUBLE PRECISION,
                    flutter_organization_index DOUBLE PRECISION,
                    flutter_baseline_dominant_hz DOUBLE PRECISION,
                    flutter_baseline_detected BOOLEAN,
                    qrs_axis_deg DOUBLE PRECISION,
                    qtc_dispersion_ms DOUBLE PRECISION,
                    electrical_alternans_detected BOOLEAN,
                    epsilon_wave_detected BOOLEAN,
                    spodick_sign_detected BOOLEAN,
                    rhythm_classification TEXT,
                    hrv_mean_nn DOUBLE PRECISION,
                    hrv_sdnn DOUBLE PRECISION,
                    hrv_rmssd DOUBLE PRECISION,
                    r_wave_inverted BOOLEAN,
                    r_peak_idx INTEGER,
                    qrs_onset INTEGER,
                    qrs_offset INTEGER,
                    p_onset INTEGER,
                    p_peak INTEGER,
                    p_offset INTEGER,
                    t_onset INTEGER,
                    t_peak INTEGER,
                    t_offset INTEGER,
                    feature_source TEXT,
                    t_peak_position INTEGER,
                    p_peak_position INTEGER,
                    rt_interval_ms DOUBLE PRECISION,
                    amplitude_mean DOUBLE PRECISION,
                    amplitude_std DOUBLE PRECISION,
                    amplitude_min DOUBLE PRECISION,
                    amplitude_max DOUBLE PRECISION,
                    peak_to_peak DOUBLE PRECISION,
                    signal_quality_score DOUBLE PRECISION,
                    is_abnormal BOOLEAN DEFAULT FALSE,
                    raw_feature_json JSONB
                )
            ''')
            
            # Ensure new columns exist for older tables
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS t_peak_position INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS p_peak_position INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS rt_interval_ms DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS pr_interval_ms DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS pr_segment_ms DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS r_wave_inverted BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS r_peak_idx INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS qrs_onset INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS qrs_offset INTEGER"
            )
            # NeuroKit2 integration columns
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS st_segment_ms DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS t_wave_width_ms DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS p_wave_inverted BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS p_onset INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS p_peak INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS p_offset INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS t_onset INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS t_peak INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS t_offset INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS feature_source TEXT"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS qrs_voltage DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS q_amplitude DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS r_amplitude DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS s_amplitude DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS q_peak INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS s_peak INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS qtc_bazett DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS qtc_fridericia DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS t_wave_amplitude DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS t_wave_polarity TEXT"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS tpeak_tend_interval_ms DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS p_wave_amplitude DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS p_wave_polarity TEXT"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS u_wave_detected BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS u_wave_peak INTEGER"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS u_wave_amplitude DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS delta_wave_detected BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS delta_wave_slope DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS flutter_baseline_power DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS flutter_baseline_dominant_hz DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS flutter_baseline_detected BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS flutter_organization_index DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS qrs_axis_deg DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS qtc_dispersion_ms DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS electrical_alternans_detected BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS epsilon_wave_detected BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS spodick_sign_detected BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS rhythm_classification TEXT"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS hrv_mean_nn DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS hrv_sdnn DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS hrv_rmssd DOUBLE PRECISION"
            )
            cursor.execute(
                "ALTER TABLE beat_features ADD COLUMN IF NOT EXISTS context_samples DOUBLE PRECISION[]"
            )
            
            # Index for fast retrieval of recent beats by session
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_beat_session_time 
                ON beat_features (session_id, timestamp DESC)
            ''')
            
            # Table for detected rhythm events
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rhythm_events (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                
                    event_start_time TIMESTAMP NOT NULL,
                    event_end_time TIMESTAMP NOT NULL,
                
                    severity TEXT DEFAULT 'unknown',
                
                    is_active BOOLEAN DEFAULT TRUE,
                
                    metadata_json JSONB
                )
            ''')

            # Migration for rhythm_events columns
            cursor.execute("SELECT data_type FROM information_schema.columns WHERE table_name = 'rhythm_events' AND column_name = 'event_start_time'")
            row = cursor.fetchone()
            if row and 'double' in row[0].lower():
                logger.info("Migrating rhythm_events start/end times to TIMESTAMP...")
                cursor.execute("ALTER TABLE rhythm_events ALTER COLUMN event_start_time TYPE TIMESTAMP USING TO_TIMESTAMP(event_start_time)")
                cursor.execute("ALTER TABLE rhythm_events ALTER COLUMN event_end_time TYPE TIMESTAMP USING TO_TIMESTAMP(event_end_time)")
            
            cursor.execute("ALTER TABLE rhythm_events DROP COLUMN IF EXISTS last_update_time")
            cursor.execute("ALTER TABLE rhythm_events DROP COLUMN IF EXISTS last_update_time_time") # possible typo in previous versions? No, the user said last update.
            
            # Index for retrieving events by session
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_rhythm_event_session 
                ON rhythm_events (session_id, event_start_time)
            ''')
            # =====================================================
            # KNOWLEDGE CHUNKS TABLE
            # =====================================================

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_chunks (

                id BIGSERIAL PRIMARY KEY,

                chunk_id TEXT UNIQUE NOT NULL,

                condition_id TEXT NOT NULL,

                condition_name TEXT NOT NULL,

                category TEXT,

                section TEXT NOT NULL,

                content TEXT NOT NULL,

                retrieval_tags JSONB,

                source_provenance JSONB,

                metadata JSONB,

                embedding VECTOR(384),

                created_at TIMESTAMP DEFAULT NOW()
            )
            ''')

            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_knowledge_condition
            ON knowledge_chunks(condition_id)
            ''')

            # Vector similarity index
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_knowledge_embedding
            ON knowledge_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            ''')

            # =====================================================
            # PDF SEMANTIC RAG CHUNKS
            # =====================================================

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS pdf_knowledge_chunks (

                id BIGSERIAL PRIMARY KEY,

                document_name TEXT NOT NULL,

                chunk_id TEXT UNIQUE NOT NULL,

                page_start INTEGER,

                page_end INTEGER,

                chunk_index INTEGER,

                section_hint TEXT,

                content TEXT NOT NULL,

                metadata_json JSONB,

                embedding VECTOR(384),

                created_at TIMESTAMP DEFAULT NOW()
            )
            ''')

            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_pdf_knowledge_document
            ON pdf_knowledge_chunks(document_name)
            ''')

            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_pdf_knowledge_embedding
            ON pdf_knowledge_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            ''')

            # =====================================================
            # PATIENTS METADATA
            # =====================================================

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                age INTEGER,
                sex TEXT,
                smoking_status TEXT,
                comorbidities JSONB,
                dynamic_symptoms JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            ''')

            # =====================================================
            # AGENT INTERACTIONS
            # =====================================================

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_interactions (

                id BIGSERIAL PRIMARY KEY,

                patient_id TEXT NOT NULL,

                session_id TEXT,

                event_type TEXT,

                prompt_json JSONB,

                retrieved_chunks JSONB,

                response_json JSONB,

                created_at TIMESTAMP DEFAULT NOW()
            )
            ''')
    def search_knowledge(self, embedding, top_k=8):

        embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        embedding_str = "[" + ",".join(map(str, embedding_list)) + "]"

        with self._get_connection() as conn:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            cursor.execute(
                """
                SELECT
                    chunk_id,
                    condition_id,
                    condition_name,
                    section,
                    content,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM knowledge_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    embedding_str,
                    embedding_str,
                    top_k
                )
            )

            return cursor.fetchall()

    def count_pdf_knowledge_chunks(self, document_name: Optional[str] = None) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if document_name:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM pdf_knowledge_chunks
                    WHERE document_name = %s
                    """,
                    (document_name,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM pdf_knowledge_chunks")
            return int(cursor.fetchone()[0])

    def insert_pdf_knowledge_chunks(
        self,
        chunks: List[Dict[str, Any]],
        replace_document: bool = False
    ) -> int:
        if not chunks:
            return 0

        document_name = chunks[0].get("document_name")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            if replace_document and document_name:
                cursor.execute(
                    """
                    DELETE FROM pdf_knowledge_chunks
                    WHERE document_name = %s
                    """,
                    (document_name,)
                )

            inserted = 0
            for chunk in chunks:
                embedding = chunk.get("embedding")
                embedding_str = (
                    "[" + ",".join(map(str, embedding)) + "]"
                    if embedding is not None
                    else None
                )
                metadata_json = chunk.get("metadata_json") or {}
                if isinstance(metadata_json, dict):
                    metadata_json = json.dumps(metadata_json)

                cursor.execute(
                    """
                    INSERT INTO pdf_knowledge_chunks (
                        document_name,
                        chunk_id,
                        page_start,
                        page_end,
                        chunk_index,
                        section_hint,
                        content,
                        metadata_json,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (chunk_id)
                    DO UPDATE SET
                        page_start = EXCLUDED.page_start,
                        page_end = EXCLUDED.page_end,
                        chunk_index = EXCLUDED.chunk_index,
                        section_hint = EXCLUDED.section_hint,
                        content = EXCLUDED.content,
                        metadata_json = EXCLUDED.metadata_json,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        chunk.get("document_name"),
                        chunk.get("chunk_id"),
                        chunk.get("page_start"),
                        chunk.get("page_end"),
                        chunk.get("chunk_index"),
                        chunk.get("section_hint"),
                        chunk.get("content"),
                        metadata_json,
                        embedding_str,
                    )
                )
                inserted += 1

            cursor.execute("ANALYZE pdf_knowledge_chunks")
            return inserted

    def search_pdf_knowledge(
        self,
        embedding,
        top_k: int = 3,
        document_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        embedding_str = "[" + ",".join(map(str, embedding_list)) + "]"

        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SET LOCAL ivfflat.probes = 100")

            if document_name:
                cursor.execute(
                    """
                    SELECT
                        chunk_id,
                        document_name,
                        page_start,
                        page_end,
                        chunk_index,
                        section_hint,
                        content,
                        metadata_json,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM pdf_knowledge_chunks
                    WHERE document_name = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        embedding_str,
                        document_name,
                        embedding_str,
                        top_k
                    )
                )
            else:
                cursor.execute(
                    """
                    SELECT
                        chunk_id,
                        document_name,
                        page_start,
                        page_end,
                        chunk_index,
                        section_hint,
                        content,
                        metadata_json,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM pdf_knowledge_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        embedding_str,
                        embedding_str,
                        top_k
                    )
                )

            return cursor.fetchall()
    def insert_beat(self, beat_data):

        raw_json = beat_data.get("raw_feature_json")

        if isinstance(raw_json, dict):
            raw_json = json.dumps(raw_json)

        with self._get_connection() as conn:

            cursor = conn.cursor()
            columns = [
                "session_id", "timestamp", "beat_index",
                "predicted_label", "prediction_confidence",
                "rr_interval", "heart_rate",
                "qrs_width", "qrs_voltage", "q_amplitude", "r_amplitude", "s_amplitude",
                "q_peak", "s_peak", "r_peak_idx", "qrs_onset", "qrs_offset",
                "qt_interval", "qtc", "qtc_bazett", "qtc_fridericia",
                "st_deviation", "st_segment_ms",
                "t_wave_min", "t_wave_inverted", "t_wave_width_ms",
                "t_wave_amplitude", "t_wave_polarity", "tpeak_tend_interval_ms",
                "p_wave_detected", "p_wave_width_ms", "p_wave_prominence",
                "p_wave_inverted", "p_wave_amplitude", "p_wave_polarity",
                "pr_interval_ms", "pr_segment_ms",
                "u_wave_detected", "u_wave_peak", "u_wave_amplitude",
                "delta_wave_detected", "delta_wave_slope",
                "flutter_baseline_power", "flutter_baseline_dominant_hz",
                "flutter_baseline_detected", "flutter_organization_index",
                "qrs_axis_deg", "qtc_dispersion_ms",
                "electrical_alternans_detected", "epsilon_wave_detected",
                "spodick_sign_detected", "rhythm_classification",
                "hrv_mean_nn", "hrv_sdnn", "hrv_rmssd",
                "r_wave_inverted",
                "p_onset", "p_peak", "p_offset", "t_onset", "t_peak", "t_offset",
                "feature_source",
                "t_peak_position", "p_peak_position", "rt_interval_ms",
                "amplitude_mean", "amplitude_std", "amplitude_min", "amplitude_max",
                "peak_to_peak", "signal_quality_score", "is_abnormal",
                "raw_feature_json",
                "context_samples",
            ]

            values = []
            for column in columns:
                if column == "raw_feature_json":
                    values.append(raw_json)
                elif column == "is_abnormal":
                    values.append(beat_data.get(column, False))
                else:
                    values.append(beat_data.get(column))

            placeholders = [
                "%s::jsonb" if column == "raw_feature_json" else "%s"
                for column in columns
            ]
            query = f"""
                INSERT INTO beat_features ({", ".join(columns)})
                VALUES ({", ".join(placeholders)})
                RETURNING id
            """

            cursor.execute(query, values)

            return cursor.fetchone()[0]
    def get_recent_events(
        self,
        session_id,
        limit=10
    ):

        with self._get_connection() as conn:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            cursor.execute(
                """
                SELECT *
                FROM rhythm_events
                WHERE session_id = %s
                ORDER BY event_end_time DESC
                LIMIT %s
                """,
                (
                    session_id,
                    limit
                )
            )

            return cursor.fetchall()
    def insert_rhythm_event(self, event_data: Dict[str, Any]) -> int:
        """
        Inserts a detected rhythm or pattern event into the database.
        Returns the inserted row ID.
        """
        meta_json = event_data.get('metadata_json')
        if isinstance(meta_json, dict):
            meta_json = json.dumps(meta_json)

        # Convert float timestamps to datetime
        start_time = event_data['event_start_time']
        if isinstance(start_time, (int, float)):
            start_time = datetime.fromtimestamp(start_time)
            
        end_time = event_data['event_end_time']
        if isinstance(end_time, (int, float)):
            end_time = datetime.fromtimestamp(end_time)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO rhythm_events (
                    session_id, event_type, event_start_time, event_end_time, 
                    severity, metadata_json
                ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                RETURNING id
            ''', (
                event_data['session_id'],
                event_data['event_type'],
                start_time,
                end_time,
                event_data.get('severity', 'unknown'),
                meta_json
            ))
            return cursor.fetchone()[0]
    def register_patient(self, patient_id: str, age: int, sex: str, smoking_status: str, comorbidities: list):
        """Registers a new patient with fixed metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO patients (patient_id, age, sex, smoking_status, comorbidities, dynamic_symptoms)
                VALUES (%s, %s, %s, %s, %s::jsonb, '{}'::jsonb)
                ON CONFLICT (patient_id) DO UPDATE SET
                    age = EXCLUDED.age,
                    sex = EXCLUDED.sex,
                    smoking_status = EXCLUDED.smoking_status,
                    comorbidities = EXCLUDED.comorbidities,
                    updated_at = NOW()
            ''', (patient_id, age, sex, smoking_status, json.dumps(comorbidities)))

    def update_dynamic_symptom(self, patient_id: str, symptom_key: str, answer: str):
        """Updates or adds a dynamic symptom answer to the patient's record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE patients 
                SET dynamic_symptoms = jsonb_set(
                        dynamic_symptoms, 
                        array[%s], 
                        %s::jsonb, 
                        true
                    ),
                    updated_at = NOW()
                WHERE patient_id = %s
            ''', (symptom_key, json.dumps(answer), patient_id))

    def get_patient_metadata(self, patient_id: str) -> dict:
        """Retrieves unified patient metadata (fixed + dynamic)."""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute('''
                SELECT age, sex, smoking_status, comorbidities, dynamic_symptoms
                FROM patients
                WHERE patient_id = %s
            ''', (patient_id,))
            row = cursor.fetchone()
            if not row:
                return {}
            
            # Combine into a single dictionary format expected by the pipeline
            return {
                "age": row["age"],
                "sex": row["sex"],
                "smoking_status": row["smoking_status"],
                "comorbidities": row["comorbidities"] or [],
                "dynamic_symptoms": row["dynamic_symptoms"] or {}
            }
    def insert_agent_interaction(
        self,
        patient_id,
        session_id,
        event_type,
        prompt_json,
        retrieved_chunks,
        response_json
    ):

        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO agent_interactions (

                    patient_id,
                    session_id,
                    event_type,

                    prompt_json,
                    retrieved_chunks,
                    response_json

                )

                VALUES (

                    %s,%s,%s,
                    %s::jsonb,
                    %s::jsonb,
                    %s::jsonb

                )

                RETURNING id
                """,
                (
                    patient_id,
                    session_id,
                    event_type,

                    json.dumps(prompt_json),

                    json.dumps(retrieved_chunks),

                    json.dumps(response_json)
                )
            )

            return cursor.fetchone()[0]
    def get_recent_beats(self, session_id: str, limit: int = 50) -> List[Dict]:
        """
        Retrieves the most recent `limit` beats for a given session, 
        ordered chronologically (oldest to newest within that window).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute('''
                SELECT * FROM (
                    SELECT * FROM beat_features 
                    WHERE session_id = %s 
                    ORDER BY timestamp DESC, beat_index DESC
                    LIMIT %s
                ) sub
                ORDER BY timestamp ASC, beat_index ASC
            ''', (session_id, limit))
            return cursor.fetchall()

    def get_session_events(self, session_id: str) -> List[Dict]:
        """
        Retrieves all rhythm events for a session, ordered by start time.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute('''
                SELECT * FROM rhythm_events 
                WHERE session_id = %s 
                ORDER BY event_start_time ASC
            ''', (session_id,))
            return cursor.fetchall()
    def get_active_event(
        self,
        session_id: str,
        event_type: str
    ):    

        with self._get_connection() as conn:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            cursor.execute(
                '''
                SELECT *
                FROM rhythm_events
            WHERE session_id = %s
            AND event_type = %s
            AND is_active = TRUE
            ORDER BY event_end_time DESC
            LIMIT 1
            ''',
            (session_id, event_type)
        )
        return cursor.fetchone()
    def update_active_event(
        self,
        event_id: int,
        new_end_time: float,
        metadata_json=None,
        severity=None
    ):
        # Convert float timestamp to datetime
        if isinstance(new_end_time, (int, float)):
            new_end_time = datetime.fromtimestamp(new_end_time)

        with self._get_connection() as conn:

            cursor = conn.cursor()

            if isinstance(metadata_json, dict):
                metadata_json = json.dumps(metadata_json)

            cursor.execute(
                '''
                UPDATE rhythm_events
                SET
                    event_end_time = %s,
                    metadata_json = COALESCE(%s::jsonb, metadata_json),
                    severity = COALESCE(%s, severity)
                WHERE id = %s
                ''',
                (
                    new_end_time,
                    metadata_json,
                    severity,
                    event_id
                )
            )
    def close_all_active_events(self) -> int:
        """
        Close all active rhythm events across all sessions.
        This is called at pipeline init to prevent stale events from
        blocking new event creation via process_event_with_cooldown().
        Returns the number of events closed.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE rhythm_events
                SET is_active = FALSE
                WHERE is_active = TRUE
                """
            )
            return cursor.rowcount

    def close_event(
        self,
        event_id: int,
        close_time: float
    ):
        # Convert float timestamp to datetime
        if isinstance(close_time, (int, float)):
            close_time = datetime.fromtimestamp(close_time)

        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute(
                '''
                UPDATE rhythm_events
                SET
                    is_active = FALSE,
                    event_end_time = %s
                WHERE id = %s
                ''',
                (
                    close_time,
                    event_id
                )
            )
    def count_pvcs_last_24h(self, session_id: str) -> int:
        """
        Count PVC beats in the past 24 hours for a given session.
        PVC label is 2 (LABEL_V) in the AAMI standard mapping.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM beat_features
                WHERE session_id = %s
                  AND predicted_label = 2
                  AND timestamp >= EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours')
            ''', (session_id,))
            return int(cursor.fetchone()[0])

    def count_pvcs_lbbb_last_24h(self, session_id: str) -> int:
        """
        Count PVC beats with LBBB morphology in the past 24 hours.
        LBBB morphology is inferred from qrs_width >= 120 ms and qrs_axis_deg <= -30.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM beat_features
                WHERE session_id = %s
                  AND predicted_label = 2
                  AND qrs_width >= 120
                  AND (qrs_axis_deg IS NULL OR qrs_axis_deg <= -30)
                  AND timestamp >= EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours')
            ''', (session_id,))
            return int(cursor.fetchone()[0])

    def count_vt_episodes_last_24h(self, session_id: str) -> int:
        """
        Count VT_RUN events in the past 24 hours from rhythm_events.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM rhythm_events
                WHERE session_id = %s
                  AND event_type = 'VT_RUN'
                  AND event_start_time >= NOW() - INTERVAL '24 hours'
            ''', (session_id,))
            return int(cursor.fetchone()[0])

    def count_prior_arvc_epsilon_windows(self, session_id: str) -> int:
        """
        Count prior monitoring windows where epsilon wave was detected.
        Uses rhythm_events with DISEASE_ARVC type as a proxy.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM rhythm_events
                WHERE session_id = %s
                  AND event_type = 'DISEASE_ARVC'
            ''', (session_id,))
            return int(cursor.fetchone()[0])

    def count_prior_arvc_t_inversion_windows(self, session_id: str) -> int:
        """
        Count prior monitoring windows where persistent T-wave inversion
        was detected in the context of ARVC screening.
        Uses rhythm_events with DISEASE_ARVC type and T-inversion metadata.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM rhythm_events
                WHERE session_id = %s
                  AND event_type = 'DISEASE_ARVC'
                  AND metadata_json->>'t_inversion_fraction' IS NOT NULL
                  AND (metadata_json->>'t_inversion_fraction')::float >= 0.70
            ''', (session_id,))
            return int(cursor.fetchone()[0])

    def get_max_arvc_ecg_tfc_score(self, session_id: str) -> int:
        """
        Get the highest ECG-partial TFC score ever recorded for this session.
        Queries metadata_json from DISEASE_ARVC events for the tfc_score field.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COALESCE(MAX(
                    (metadata_json->>'tfc_score')::int
                ), 0) FROM rhythm_events
                WHERE session_id = %s
                  AND event_type = 'DISEASE_ARVC'
                  AND metadata_json->>'tfc_score' IS NOT NULL
            ''', (session_id,))
            return int(cursor.fetchone()[0])

    def get_condition_sections(
       self,
       condition_id,
       sections=None
    ):  

        with self._get_connection() as conn:

            cursor = conn.cursor(
                cursor_factory=RealDictCursor
            )

            if sections:

                cursor.execute(
                    """
                    SELECT
                        chunk_id,
                        condition_id,
                        condition_name,
                        section,
                        content
                    FROM knowledge_chunks
                    WHERE condition_id = %s
                    AND section = ANY(%s)
                    """,
                    (
                        condition_id,
                        sections
                    )
                )

            else:

                cursor.execute(
                    """
                    SELECT
                        chunk_id,
                        condition_id,
                        condition_name,
                        section,
                        content
                    FROM knowledge_chunks
                    WHERE condition_id = %s
                    """,
                    (
                        condition_id,
                    )
                )

            return cursor.fetchall()
    def close(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()