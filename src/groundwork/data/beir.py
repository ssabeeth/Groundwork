"""Loading BEIR datasets from their published zip archives.

BEIR ships each dataset as ``corpus.jsonl``, ``queries.jsonl`` and ``qrels/<split>.tsv``.
The corpus and query files cover every split; the qrels file selects which queries are
actually scored, so the query set is filtered down to the ids that appear in the qrels.
Evaluating over all queries instead of the judged ones is a common way to end up with
numbers that cannot be compared to anything published.

The full corpus is always retrieved against, including documents that no query judges.
"""

from __future__ import annotations

import csv
import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass(frozen=True)
class BeirDataset:
    """A BEIR dataset split ready for retrieval.

    Attributes:
        name: Dataset name, e.g. ``"scifact"``.
        split: Qrels split, e.g. ``"test"``.
        corpus: ``{doc_id: {"title": ..., "text": ...}}`` for the whole corpus.
        queries: ``{query_id: query_text}``, restricted to queries judged in this split.
        qrels: ``{query_id: {doc_id: relevance_level}}``.
    """

    name: str
    split: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def __repr__(self) -> str:
        return (
            f"BeirDataset(name={self.name!r}, split={self.split!r}, "
            f"corpus={len(self.corpus)}, queries={len(self.queries)})"
        )


def download_beir_dataset(name: str, data_dir: Path) -> Path:
    """Download and extract a BEIR dataset if it is not already present.

    Args:
        name: Dataset name as published by BEIR, e.g. ``"scifact"``.
        data_dir: Directory to hold extracted datasets.

    Returns:
        Path to the extracted dataset directory.
    """
    data_dir = Path(data_dir)
    target = data_dir / name
    if (target / "corpus.jsonl").exists():
        return target

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{name}.zip"

    if not archive.exists():
        url = BEIR_URL.format(name=name)
        logger.info("Downloading %s", url)
        with urlopen(url) as response, archive.open("wb") as handle:  # noqa: S310
            while chunk := response.read(1 << 20):
                handle.write(chunk)

    logger.info("Extracting %s", archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)

    if not (target / "corpus.jsonl").exists():
        raise FileNotFoundError(f"{target}/corpus.jsonl missing after extraction")
    return target


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_beir_dataset(
    name: str = "scifact",
    split: str = "test",
    data_dir: Path | str = "data",
    download: bool = True,
) -> BeirDataset:
    """Load a BEIR dataset, downloading it on first use.

    Args:
        name: Dataset name, e.g. ``"scifact"``, ``"trec-covid"``, ``"nfcorpus"``.
        split: Qrels split to score against.
        data_dir: Where datasets are cached.
        download: Fetch the archive if the dataset is not already extracted.

    Returns:
        The loaded dataset.

    Raises:
        FileNotFoundError: If the dataset is absent and ``download`` is False, or if
            the requested split has no qrels file.
    """
    data_dir = Path(data_dir)
    root = data_dir / name

    if not (root / "corpus.jsonl").exists():
        if not download:
            raise FileNotFoundError(
                f"{root} not found. Run with download=True or fetch it manually from "
                f"{BEIR_URL.format(name=name)}"
            )
        root = download_beir_dataset(name, data_dir)

    corpus = {
        record["_id"]: {
            "title": record.get("title", ""),
            "text": record.get("text", ""),
        }
        for record in _read_jsonl(root / "corpus.jsonl")
    }

    all_queries = {record["_id"]: record["text"] for record in _read_jsonl(root / "queries.jsonl")}

    qrels_path = root / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        available = sorted(p.stem for p in (root / "qrels").glob("*.tsv"))
        raise FileNotFoundError(f"No qrels for split {split!r}. Available: {available}")

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if header[0].strip().lower() not in {"query-id", "query_id"}:
            handle.seek(0)
            reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            query_id, doc_id, score = row[0].strip(), row[1].strip(), row[2].strip()
            level = int(float(score))
            qrels.setdefault(query_id, {})[doc_id] = level

    queries = {qid: text for qid, text in all_queries.items() if qid in qrels}
    missing = set(qrels) - set(queries)
    if missing:
        raise ValueError(f"{len(missing)} judged queries missing from queries.jsonl")

    return BeirDataset(name=name, split=split, corpus=corpus, queries=queries, qrels=qrels)
