"""FASTA reading and writing utilities using standard library only."""

from pathlib import Path


def read_fasta_sequences(path: Path) -> list[str]:
    """Read sequence strings from a FASTA file in original order."""
    sequences: list[str] = []
    current_parts: list[str] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_parts:
                sequences.append("".join(current_parts).upper())
                current_parts = []
        else:
            current_parts.append(line.upper())

    if current_parts:
        sequences.append("".join(current_parts).upper())

    return sequences


def read_fasta_entries(path: Path) -> list[tuple[str, str]]:
    """Read (header, sequence) pairs from a FASTA file."""
    entries: list[tuple[str, str]] = []
    current_header: str | None = None
    current_parts: list[str] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                entries.append((current_header, "".join(current_parts).upper()))
                current_parts = []
            current_header = line[1:].strip()
        else:
            current_parts.append(line.upper())

    if current_header is not None:
        entries.append((current_header, "".join(current_parts).upper()))

    return entries


def write_fasta_sequences(
    sequences: list[str],
    path: Path,
    header_prefix: str = "seq",
    zero_pad_width: int = 0,
) -> None:
    """Write sequences to a FASTA file with deterministic numbered headers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fasta_file:
        if zero_pad_width > 0:
            fasta_file.writelines(
                f">{header_prefix}_{index:0{zero_pad_width}d}\n{sequence}\n"
                for index, sequence in enumerate(sequences, start=1)
            )
        else:
            fasta_file.writelines(
                f">{header_prefix}_{index}\n{sequence}\n"
                for index, sequence in enumerate(sequences, start=1)
            )
