# Kaggle writeup assets

Paste-ready content and upload files for the AMP Challenge hackathon writeup
form. Start with [`WRITEUP.md`](WRITEUP.md), which has every field in the form's
order and respects the character limits.

```text
WRITEUP.md            the text to paste into each field
thumbnail.png         560 x 280 card image, upload as-is
files/library.fasta   2.3 MB, the 50,000-sequence library
files/top.fasta       4.3 KB, the ranked top-100
make_thumbnail.py     regenerates the card from the measured numbers
verify_assets.py      checks the folder is complete and self-consistent
```

Verify (from `writeup/`):

```bash
python verify_assets.py
python make_thumbnail.py
```

The two FASTA files are copies of `../generate/`, which `generate`
reproduces byte-identically. `verify_assets.py` fails if they ever drift apart,
so this folder cannot silently go stale.
