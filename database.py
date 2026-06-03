from networkx.generators import spectral_graph_forge
from networkx.generators import spectral_graph_forge
import json
import logging
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ECGDatabase:
    def __init__(self, db_config: Dict[str, str] = None, pool_min: int = 1, pool_max: int = 10):
        """
        Initialize PostgreSQL database connection.
        
        Args:
            db_config: Dictionary with database connection parameters:
                - host: Database host (default: 'localhost')
                - port: Database port (default: 5432)
                - dbname: Database name (default: 'ecg_agent_data')
                - user: Database user (default: 'postgres')
                - password: Database password (default: '')
            pool_min: Minimum number of connections in pool
            pool_max: Maximum number of connections in pool
        """
        if db_config is None:
            db_config = {}
        
        # self.db_config = {
        #     'host': db_config.get('host', 'localhost'),
        #     'port': db_config.get('port', 5432),
        #     'dbname': db_config.get('dbname', 'ecg_agent_data'),
        #     'user': db_config.get('user', 'postgres'),
        #     'password': db_config.get('password', 'AHMK@rk1')
        # }
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'ecg_agent_data',
            'user': 'postgres',
            'password': 'AHMK@rk1'
        }
        
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
                    qt_interval DOUBLE PRECISION,
                    qtc DOUBLE PRECISION,
                    st_deviation DOUBLE PRECISION,
                    t_wave_peak DOUBLE PRECISION,
                    t_wave_min DOUBLE PRECISION,
                    t_wave_inverted BOOLEAN,
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
                
                    event_start_time DOUBLE PRECISION NOT NULL,
                    event_end_time DOUBLE PRECISION NOT NULL,
                
                    severity TEXT DEFAULT 'unknown',
                
                    is_active BOOLEAN DEFAULT TRUE,
                
                    last_update_time DOUBLE PRECISION,
                
                    metadata_json JSONB
                )
            ''')
            
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
            # PATIENT MEMORY
            # =====================================================

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_memory (

                id BIGSERIAL PRIMARY KEY,

                patient_id TEXT NOT NULL,

                memory_type TEXT NOT NULL,

                memory_key TEXT NOT NULL,

                memory_value JSONB,

                confidence DOUBLE PRECISION,

                source TEXT,

                created_at TIMESTAMP DEFAULT NOW(),

                updated_at TIMESTAMP DEFAULT NOW()
            )
            ''')

            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_patient_memory
            ON patient_memory(patient_id)
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

        embedding_str = "[" + ",".join(map(str, embedding.tolist())) + "]"

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
    def insert_beat(self, beat_data):

        raw_json = beat_data.get("raw_feature_json")

        if isinstance(raw_json, dict):
            raw_json = json.dumps(raw_json)

        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO beat_features (

                    session_id,
                    timestamp,
                    beat_index,

                    predicted_label,
                    prediction_confidence,

                    rr_interval,
                    heart_rate,
                    qrs_width,

                    amplitude_mean,
                    amplitude_std,
                    amplitude_min,
                    amplitude_max,

                    peak_to_peak,

                    signal_quality_score,

                    is_abnormal,

                    raw_feature_json,

                    qt_interval,
                    qtc,

                    st_deviation,

                    t_wave_peak,
                    t_wave_min,
                    t_wave_inverted

                )

                VALUES (

                    %s,%s,%s,
                    %s,%s,
                    %s,%s,%s,
                    %s,%s,%s,%s,
                    %s,
                    %s,
                    %s,
                    %s::jsonb,
                    %s,%s,
                    %s,
                    %s,%s,%s

                )

                RETURNING id
                """,
                (

                    beat_data["session_id"],
                    beat_data["timestamp"],
                    beat_data["beat_index"],

                    beat_data.get("predicted_label"),
                    beat_data.get("prediction_confidence"),

                    beat_data.get("rr_interval"),
                    beat_data.get("heart_rate"),
                    beat_data.get("qrs_width"),

                    beat_data.get("amplitude_mean"),
                    beat_data.get("amplitude_std"),
                    beat_data.get("amplitude_min"),
                    beat_data.get("amplitude_max"),

                    beat_data.get("peak_to_peak"),

                    beat_data.get("signal_quality_score"),

                    beat_data.get("is_abnormal", False),

                    raw_json,

                    beat_data.get("qt_interval"),
                    beat_data.get("qtc"),

                    beat_data.get("st_deviation"),

                    beat_data.get("t_wave_peak"),
                    beat_data.get("t_wave_min"),
                    beat_data.get("t_wave_inverted")

                )
            )

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
                event_data['event_start_time'],
                event_data['event_end_time'],
                event_data.get('severity', 'unknown'),
                meta_json
            ))
            return cursor.fetchone()[0]
    def get_patient_memory(
        self,
         patient_id
     ):       
        """
        Retrieves all memory entries for a given patient.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute('''
                SELECT memory_type, memory_key, memory_value, confidence, source
                FROM patient_memory
                WHERE patient_id = %s
                ORDER BY updated_at DESC
            ''', (patient_id,))
            return cursor.fetchall()
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

        with self._get_connection() as conn:

            cursor = conn.cursor()

            if isinstance(metadata_json, dict):
                metadata_json = json.dumps(metadata_json)

            cursor.execute(
                '''
                UPDATE rhythm_events
                SET
                    event_end_time = %s,
                    last_update_time = %s,
                    metadata_json = COALESCE(%s::jsonb, metadata_json),
                    severity = COALESCE(%s, severity)
                WHERE id = %s
                ''',
                (
                    new_end_time,
                    new_end_time,
                    metadata_json,
                    severity,
                    event_id
                )
            )
    def close_event(
        self,
        event_id: int,
        close_time: float
    ):

        with self._get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute(
                '''
                UPDATE rhythm_events
                SET
                    is_active = FALSE,
                    event_end_time = %s,
                    last_update_time = %s
                WHERE id = %s
                ''',
                (
                    close_time,
                    close_time,
                    event_id
                )
            )
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