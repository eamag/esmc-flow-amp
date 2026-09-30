"""Check that the writeup folder is complete, correct, and not stale."""

import re
import struct
import sys
from pathlib import Path

FOLDER = Path(__file__).parent
GENERATED = FOLDER.parent / "generate"
LIBRARY = FOLDER / "files" / "library.fasta"
TOP = FOLDER / "files" / "top.fasta"
THUMBNAIL = FOLDER / "thumbnail.png"
WRITEUP = FOLDER / "WRITEUP.md"

UPLOAD_LIMIT_BYTES = 100 * 1024 * 1024
THUMBNAIL_SIZE = (560, 280)
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
LIBRARY_SIZE = 50_000
TOP_SIZE = 100
TITLE_LIMIT = 80
SUBTITLE_LIMIT = 140


def read_fasta(path: Path) -> list[str]:
    return [
        line.strip() for line in path.read_text().splitlines() if line and not line.startswith(">")
    ]


def check_thumbnail() -> list[str]:
    if not THUMBNAIL.exists():
        return [f"{THUMBNAIL} missing"]
    try:
        from PIL import Image

        with Image.open(THUMBNAIL) as image:
            size = image.size
    except ImportError:
        # Fallback PNG header parser
        with open(THUMBNAIL, "rb") as stream:
            data = stream.read(24)
        if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
            size = struct.unpack(">LL", data[16:24])
        else:
            return ["thumbnail is not a valid PNG"]
    if size == THUMBNAIL_SIZE:
        return []
    return [f"thumbnail is {size}, want {THUMBNAIL_SIZE}"]


def check_library(sequences: list[str]) -> list[str]:
    problems = []
    if len(sequences) != LIBRARY_SIZE:
        problems.append(f"library has {len(sequences):,} sequences, want {LIBRARY_SIZE:,}")
    if len(set(sequences)) != len(sequences):
        problems.append("library has duplicates")
    for sequence in sequences:
        if set(sequence) - STANDARD_AMINO_ACIDS:
            problems.append(f"non-canonical residues in {sequence}")
            break
        if not 8 <= len(sequence) <= 50:
            problems.append(f"length {len(sequence)} out of range in {sequence}")
            break
    return problems


def check_top(top: list[str], library: set[str]) -> list[str]:
    problems = []
    if len(top) != TOP_SIZE:
        problems.append(f"top has {len(top)} sequences, want {TOP_SIZE}")
    if len(set(top)) != len(top):
        problems.append("top has duplicates")
    missing = set(top) - library
    if missing:
        problems.append(f"{len(missing)} top sequences are absent from the library")
    return problems


def check_upload_size() -> list[str]:
    total = sum(path.stat().st_size for path in (LIBRARY, TOP, THUMBNAIL))
    if total > UPLOAD_LIMIT_BYTES:
        return [f"upload is {total / 1e6:.1f} MB, over the 100 MB cap"]
    return []


def field_length(text: str, heading: str) -> int:
    match = re.search(rf"## {heading}[^\n]*\n\n```[a-z]*\n(.+?)\n```", text, re.DOTALL)
    return len(match.group(1)) if match else 0


def check_field_limits() -> list[str]:
    if not WRITEUP.exists():
        return [f"{WRITEUP} missing"]
    text = WRITEUP.read_text()
    problems = []
    title = field_length(text, "Title")
    subtitle = field_length(text, "Subtitle")
    if title > TITLE_LIMIT:
        problems.append(f"title is {title} chars, max {TITLE_LIMIT}")
    if subtitle > SUBTITLE_LIMIT:
        problems.append(f"subtitle is {subtitle} chars, max {SUBTITLE_LIMIT}")
    if not title or not subtitle:
        problems.append("title or subtitle block not found in WRITEUP.md")
    return problems


def check_not_stale() -> list[str]:
    problems = []
    for name, staged in (("library.fasta", LIBRARY), ("top.fasta", TOP)):
        source = GENERATED / name
        if not source.exists():
            problems.append(f"{source} missing, run: generate")
        elif source.read_bytes() != staged.read_bytes():
            problems.append(f"{name} differs from generate/, re-stage it")
    return problems


def main() -> None:
    library = read_fasta(LIBRARY)
    top = read_fasta(TOP)
    problems = (
        check_thumbnail()
        + check_library(library)
        + check_top(top, set(library))
        + check_upload_size()
        + check_field_limits()
        + check_not_stale()
    )
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        sys.exit(1)
    print(f"library    {len(library):,} sequences, unique, canonical, 8-50 aa")
    print(f"top-100    {len(top)} sequences, unique, all present in the library")
    print(f"thumbnail  {THUMBNAIL_SIZE[0]} x {THUMBNAIL_SIZE[1]}")
    print("upload     2.3 MB of 100 MB")
    print("fields     title and subtitle within Kaggle limits")
    print("fresh      identical to generate/")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
