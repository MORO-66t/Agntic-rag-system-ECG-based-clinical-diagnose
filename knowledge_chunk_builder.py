import json
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"

OUTPUT_FILE = BASE_DIR / "knowledge_chunks.jsonl"
print("Current working dir:", Path.cwd())
print("Knowledge dir:", KNOWLEDGE_DIR.resolve())
def normalize_content(value):
    """
    Convert any JSON field to readable text.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "\n".join(str(x) for x in value)

    if isinstance(value, dict):

        parts = []

        for k, v in value.items():

            if isinstance(v, list):

                parts.append(
                    f"{k}: " + ", ".join(str(x) for x in v)
                )

            elif isinstance(v, dict):

                nested = []

                for nk, nv in v.items():

                    if isinstance(nv, list):
                        nested.append(
                            f"{nk}: " + ", ".join(str(x) for x in nv)
                        )

                    else:
                        nested.append(f"{nk}: {nv}")

                parts.append(
                    f"{k}: {' | '.join(nested)}"
                )

            else:
                parts.append(f"{k}: {v}")

        return "\n".join(parts)

    return str(value)


def build_chunk(
        condition_id,
        condition_name,
        category,
        section,
        content,
        retrieval_tags,
        source_provenance
):
    return {

        "chunk_id":
            f"{condition_id}_{section}",

        "condition_id":
            condition_id,

        "condition_name":
            condition_name,

        "category":
            category,

        "section":
            section,

        "content":
            content,

        "retrieval_tags":
            retrieval_tags,

        "source_provenance":
            source_provenance
    }


def process_file(json_file):

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    condition_id = data.get("condition_id")
    condition_name = data.get("display_name")
    category = data.get("category")

    retrieval_tags = data.get(
        "retrieval_tags",
        []
    )

    source_provenance = data.get(
        "source_provenance",
        []
    )

    chunks = []

    ignored_fields = {
        "condition_id",
        "display_name",
        "aliases",
        "category",
        "retrieval_tags",
        "source_provenance"
    }

    for key, value in data.items():

        if key in ignored_fields:
            continue

        content = normalize_content(value)

        if not content.strip():
            continue

        chunks.append(
            build_chunk(
                condition_id=condition_id,
                condition_name=condition_name,
                category=category,
                section=key,
                content=content,
                retrieval_tags=retrieval_tags,
                source_provenance=source_provenance
            )
        )

    return chunks


def main():

    all_chunks = []

    for json_file in KNOWLEDGE_DIR.rglob("*.json"):

        try:

            chunks = process_file(json_file)

            all_chunks.extend(chunks)

            print(
                f"[OK] {json_file.name} -> {len(chunks)} chunks"
            )

        except Exception as e:

            print(
                f"[ERROR] {json_file.name}: {e}"
            )

    with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
    ) as out:

        for chunk in all_chunks:

            out.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                )
            )

            out.write("\n")

    print(
        f"\nGenerated {len(all_chunks)} chunks"
    )

    print(
        f"Saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()