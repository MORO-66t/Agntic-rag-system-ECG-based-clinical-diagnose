import json

from sentence_transformers import SentenceTransformer

from database import ECGDatabase


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    db = ECGDatabase()

    with open(
        "knowledge_chunks.jsonl",
        "r",
        encoding="utf-8"
    ) as f:

        chunks = [
            json.loads(line)
            for line in f
        ]

    print(
        f"Loaded {len(chunks)} chunks"
    )

    with db._get_connection() as conn:

        cursor = conn.cursor()

        inserted = 0

        for chunk in chunks:

            embedding_text = f"""
            Condition: {chunk['condition_name']}

            Category: {chunk['category']}

            Section: {chunk['section']}

            Tags:
            {' '.join(chunk['retrieval_tags'])}

            Content:
            {chunk['content']}
            """

            embedding = model.encode(
                embedding_text,
                normalize_embeddings=True
            ).tolist()

            cursor.execute(
                """
                INSERT INTO knowledge_chunks (

                    chunk_id,

                    condition_id,

                    condition_name,

                    category,

                    section,

                    content,

                    retrieval_tags,

                    source_provenance,

                    metadata,

                    embedding

                )

                VALUES (

                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s

                )

                ON CONFLICT (chunk_id)
                DO NOTHING
                """,
                (
                    chunk["chunk_id"],
                    chunk["condition_id"],
                    chunk["condition_name"],
                    chunk["category"],
                    chunk["section"],
                    chunk["content"],
                    json.dumps(
                        chunk["retrieval_tags"]
                    ),
                    json.dumps(
                        chunk["source_provenance"]
                    ),
                    json.dumps({}),
                    embedding
                )
            )

            inserted += 1

            if inserted % 25 == 0:

                print(
                    f"{inserted}/{len(chunks)}"
                )

    print("DONE")


if __name__ == "__main__":
    main()
    # results = r.search(
    #     "atrial fibrillation"
    #     )       
    # print("\nsecond Results:")
    # r.search("irregular heartbeat stroke risk")