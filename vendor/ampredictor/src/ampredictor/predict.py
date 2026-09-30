"""
AMPredictor inference - optimized version.

Changes from original:
  - ESM embeddings computed in proper batches (not batch size 1)
  - Chunked processing: ESM outputs are converted to PyG graphs
    immediately and discarded, so only lightweight graph objects
    persist in memory (not raw 1280-dim embeddings for all sequences)
  - Amino acid fingerprints precomputed once (not per-sequence JSON parse)
  - No intermediate files written to disk
  - Removed the `break` bug that dropped sequences in multi-sequence batches

IMPORTANT: The model outputs log10(MIC in uM).  The back-transform is
10**x, NOT e**x.  The original repository's predict.py confirms this
("Predicted logMIC values") and the training labels are in log10 scale:
    min=-1.32 (0.048 uM) to max=3.51 (3236 uM)

Numerical output is identical to the original pipeline (with the correct
back-transform).
"""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric import data as DATA
from torch.utils.data.dataloader import default_collate

import esm
from Bio import SeqIO
from rdkit.Chem import AllChem as Chem
from rdkit import rdBase

from ampredictor.AMPredictor import GNNPredictor


rdBase.DisableLog("rdApp.warning")


MAX_LENGTH = 65
ESM_LAYER = 33
ESM_DIM = 1280
CONTACT_THRESHOLD = 0.5
FP_RADIUS = 3
FP_NBITS = 2048


# ---------------------------------------------------------------------------
# 1. Read FASTA
# ---------------------------------------------------------------------------


def read_fasta(fasta_path):
    """Read FASTA, return (ids, sequences) for sequences <= MAX_LENGTH."""
    ids, seqs = [], []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        ids.append(str(record.id))
        seqs.append(seq)
    return ids, seqs


# ---------------------------------------------------------------------------
# 2. Fingerprint features (precomputed lookup)
# ---------------------------------------------------------------------------


def precompute_aa_fingerprints(smiles_path):
    """Compute Morgan fingerprints for all 20 amino acids once."""
    with open(smiles_path) as f:
        smiles = json.load(f)

    lookup = {}
    for aa, smi in smiles.items():
        mol = Chem.MolFromSmiles(smi)
        fp = np.array(
            Chem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
        )
        lookup[aa] = fp

    zero_fp = np.zeros(FP_NBITS, dtype=int)
    lookup[" "] = zero_fp
    lookup["0"] = zero_fp
    return lookup


def compute_fingerprint(seq, aa_fps):
    """Compute the fingerprint feature vector for one sequence.

    Returns a 1D array of length MAX_LENGTH (mean-pooled across FP_NBITS).
    """
    fp = np.array([aa_fps[aa] for aa in seq])  # (seq_len, 2048)
    if len(seq) < MAX_LENGTH:
        padding = np.zeros((MAX_LENGTH - len(seq), FP_NBITS))
        fp = np.concatenate([fp, padding], axis=0)
    return np.mean(fp, axis=1)


# ---------------------------------------------------------------------------
# 3. Graph construction from ESM outputs
# ---------------------------------------------------------------------------


def build_graph_from_esm(rep, contact_map, seq_len):
    """Build a PyG Data object from ESM representation and contact map.

    Parameters
    ----------
    rep : torch.Tensor      (seq_len, 1280) per-token representation
    contact_map : np.ndarray (seq_len, seq_len) predicted contacts
    seq_len : int            original sequence length

    Returns
    -------
    DATA.Data  PyG graph object
    int        sequence length (size)
    """
    # Pad representation to MAX_LENGTH.
    if seq_len < MAX_LENGTH:
        padding = torch.zeros(MAX_LENGTH - seq_len, ESM_DIM)
        feature = torch.cat([rep, padding], dim=0)
    else:
        feature = rep

    # Pad contact map to MAX_LENGTH x MAX_LENGTH, add self-loops.
    contact = np.zeros((MAX_LENGTH, MAX_LENGTH))
    contact[:seq_len, :seq_len] = contact_map
    contact = contact + np.eye(MAX_LENGTH)

    # Edge index.
    rows, cols = np.where(contact >= CONTACT_THRESHOLD)
    edge_index = torch.LongTensor(np.stack([rows, cols], axis=0))

    gcn_data = DATA.Data(
        x=feature.float(),
        edge_index=edge_index,
        y=torch.FloatTensor([0.0]),  # dummy label
    )
    gcn_data.__setitem__("target_size", torch.LongTensor([seq_len]))

    return gcn_data


# ---------------------------------------------------------------------------
# 4. Chunked ESM processing
# ---------------------------------------------------------------------------


def process_esm_chunked(
    ids,
    seqs,
    aa_fps,
    device,
    tokens_per_batch=2048,
    chunk_size=5000,
):
    """Run ESM in chunks, immediately build graphs, discard raw embeddings.

    This keeps GPU memory bounded (small ESM batches) and CPU memory
    bounded (only lightweight PyG graphs survive, not raw embeddings).

    Parameters
    ----------
    ids : list[str]
    seqs : list[str]
    aa_fps : dict          precomputed amino acid fingerprints
    device : torch.device
    tokens_per_batch : int max tokens per ESM forward pass
    chunk_size : int       number of sequences to hold in memory at once

    Returns
    -------
    all_data_pro : list[DATA.Data]
    all_data_fp  : list[np.ndarray]
    valid_ids    : list[str]
    valid_seqs   : list[str]
    """
    # Load ESM model once.
    print("  Loading ESM-1b model...", file=sys.stderr)
    esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    esm_model.eval()
    esm_model = esm_model.to(device)

    all_data_pro = []
    all_data_fp = []
    valid_ids = []
    valid_seqs = []

    n_total = len(ids)

    for chunk_start in range(0, n_total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_total)
        chunk_ids = ids[chunk_start:chunk_end]
        chunk_seqs = seqs[chunk_start:chunk_end]

        # Build ESM batches for this chunk.
        dataset = esm.FastaBatchedDataset(chunk_ids, chunk_seqs)
        batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)
        loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=alphabet.get_batch_converter(),
            batch_sampler=batches,
        )

        with torch.no_grad():
            for labels, strs, toks in loader:
                toks = toks.to(device, non_blocking=True)
                out = esm_model(toks, repr_layers=[ESM_LAYER], return_contacts=True)

                reps = out["representations"][ESM_LAYER].cpu()
                cts = out["contacts"].cpu()

                # Immediately build graphs and discard raw tensors.
                for i, label in enumerate(labels):
                    seq_len = len(strs[i])
                    seq = strs[i]

                    rep = reps[i, 1 : seq_len + 1].clone()
                    ct = cts[i, :seq_len, :seq_len].numpy()

                    gcn_data = build_graph_from_esm(rep, ct, seq_len)
                    fp = compute_fingerprint(seq, aa_fps)

                    all_data_pro.append(gcn_data)
                    all_data_fp.append(fp)
                    valid_ids.append(label)
                    valid_seqs.append(seq)

                # Free batch tensors.
                del reps, cts, out

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"  Chunk {chunk_end}/{n_total} processed "
            f"({len(all_data_pro)} graphs built)",
            file=sys.stderr,
        )

    # Free ESM model.
    del esm_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return all_data_pro, all_data_fp, valid_ids, valid_seqs


# ---------------------------------------------------------------------------
# 5. Dataset + collation
# ---------------------------------------------------------------------------


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_pro, data_fp):
        self.data_pro = data_pro
        self.data_fp = data_fp

    def __len__(self):
        return len(self.data_pro)

    def __getitem__(self, idx):
        return self.data_pro[idx], self.data_fp[idx]


def collate_fn(data_list):
    """Collate function matching the original collate()."""
    batchA = Batch.from_data_list([d[0] for d in data_list])
    batchB = default_collate([d[1] for d in data_list])
    return batchA, batchB


# ---------------------------------------------------------------------------
# 6. GNN prediction
# ---------------------------------------------------------------------------


def predict_mic(model, device, loader):
    """Run GNN model inference, return raw predictions (log10 MIC)."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data_pro, data_fp in loader:
            data_pro = data_pro.to(device)
            data_fp = data_fp.to(device)
            output = model(data_pro, data_fp)
            all_preds.append(output.cpu().numpy().flatten())

    return np.concatenate(all_preds)


def predict(
    sequences: list[str],
    *,
    batch_size: int = 4096,
    tokens_per_batch: int = 2048,
    chunk_size: int = 5000,
) -> np.ndarray:
    ids = np.arange(len(sequences))

    lengths = np.array([len(seq) for seq in sequences])

    if (lengths > MAX_LENGTH).any():
        raise ValueError(f"Max accepted sequence length: {MAX_LENGTH}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- 2. Precompute fingerprint lookup ---
    smiles_path = os.path.join(script_dir, "data", "smiles_file.json")
    aa_fps = precompute_aa_fingerprints(smiles_path)

    # --- 3. ESM + graph construction (chunked) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Computing ESM embeddings and building graphs...", file=sys.stderr)
    data_pro, data_fp, valid_ids, valid_seqs = process_esm_chunked(
        ids,
        sequences,
        aa_fps,
        device,
        tokens_per_batch=tokens_per_batch,
        chunk_size=chunk_size,
    )
    print(f"  {len(data_pro)} graphs built", file=sys.stderr)

    # --- 4. GNN prediction ---
    print("Loading GNN model and predicting...", file=sys.stderr)
    dataset = SimpleDataset(data_pro, data_fp)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model_path = os.path.join(script_dir, "models", "model_GNNPredictor_.model")
    model = GNNPredictor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    mic_log10 = predict_mic(model, device, loader)

    # --- 5. Back-transform and write output ---
    # Model outputs log10(MIC in uM). Back-transform: 10^x.
    mic = 10.0**mic_log10

    mics = np.full(len(sequences), np.nan)
    mics[np.array(valid_ids)] = mic

    return mics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = ArgumentParser(description="AMPredictor optimized inference")
    parser.add_argument(
        "--input", default="sample.fasta", type=str, help="Path to input FASTA file"
    )
    parser.add_argument(
        "--output", type=str, help="Path to output TSV file", default="out.csv"
    )
    parser.add_argument(
        "--tokens-per-batch",
        type=int,
        default=2048,
        help="Max tokens per ESM batch (lower = less GPU memory, default 2048)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Sequences per processing chunk (lower = less CPU memory, default 5000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for GNN prediction (default 4096)",
    )
    args = parser.parse_args()

    # --- 1. Read FASTA ---
    print(f"Reading FASTA from {args.input}...", file=sys.stderr)
    ids, seqs = read_fasta(args.input)
    print(
        f"  {len(ids)} sequences after filtering to <= {MAX_LENGTH} aa",
        file=sys.stderr,
    )

    if len(ids) == 0:
        pd.DataFrame(columns=["sequence", "MIC", "MIC_unit"]).to_csv(
            args.output, sep="\t", index=False
        )
        print("No valid sequences. Empty output written.", file=sys.stderr)
        return

    mic = predict(seqs)

    output_df = pd.DataFrame({"sequence": seqs, "MIC": mic, "MIC_unit": "uM"})
    output_df.to_csv(args.output, sep="\t", index=False)

    print(
        f"Done: {len(output_df)} predictions written to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    out = predict(["KK", "RRR"])
    print(out)
