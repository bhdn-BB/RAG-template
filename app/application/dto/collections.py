from dataclasses import dataclass


@dataclass(frozen=True)
class CollectionDTO:
    name: str
    chunk_count: int = 0
    document_count: int | None = None  # computed only for single-collection reads
