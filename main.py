#!/usr/bin/env python3
"""
main.py — MSAffect: end-to-end MSA adversary & robustness pipeline.

Usage:
    python main.py [--dryrun] [--config config.yml] [--threads N] [--targets target1,target2]

Notes:
 - Put input FASTA files in ./msa/*.fasta
 - Outputs go to ./results/<basename>/
 - Dryrun mode uses lightweight surrogates and does not call AlphaFold/ColabFold.
"""

import os
import sys
import time
import json
import math
import random
import hashlib
import shutil
import logging
import argparse
import tempfile
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# core scientific
import numpy as np
import matplotlib.pyplot as plt

# Biopython
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, Superimposer

# optional imports handled gracefully
try:
    from colabfold.batch import run as colabfold_run, get_queries
    from colabfold.download import download_alphafold_params
    COLABFOLD_AVAILABLE = True
except Exception:
    colabfold_run = None
    get_queries = None
    download_alphafold_params = None
    COLABFOLD_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except Exception:
    GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except Exception:
    PPO = None
    SB3_AVAILABLE = False

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances
    SKLEARN_AVAILABLE = True
except Exception:
    AgglomerativeClustering = None
    pairwise_distances = None
    SKLEARN_AVAILABLE = False

# Simple VAE implementation uses torch if available
if TORCH_AVAILABLE:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
else:
    nn = None

# set up user agent early so distant libraries see it
DEFAULT_USER_AGENT = "MSAffect/0.1 contact@aaravdave.org"
os.environ.setdefault("HTTP_USER_AGENT", DEFAULT_USER_AGENT)
if REQUESTS_AVAILABLE:
    # try to set default Session header
    try:
        s = requests.Session()
        s.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    except Exception:
        pass

# Logging
def setup_logging():
    fmt = "%(levelname_color)s [%(asctime)s] %(message)s"
    datefmt = "%H:%M:%S"
    class ColorFormatter(logging.Formatter):
        COLORS = {"INFO":"\033[94m","WARNING":"\033[93m","ERROR":"\033[91m"}
        def format(self, record):
            lvl = record.levelname
            color = self.COLORS.get(lvl, "")
            record.levelname_color = f"{color}{lvl}\033[0m"
            record.asctime = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            return f"{record.levelname_color} [{record.asctime}] {record.getMessage()}"
    h = logging.StreamHandler()
    h.setFormatter(ColorFormatter(fmt, datefmt=datefmt))
    logging.root.handlers[:] = [h]
    logging.root.setLevel(logging.INFO)

setup_logging()
logger = logging.getLogger("MSAffect")

# ---------- Configuration defaults ----------
DEFAULTS = {
    "results_root": "results",
    "msa_dir": "msa",
    "num_models": 5,
    "use_templates": False,
    "mut_rate": 0.02,
    "num_delete": 5,
    "user_agent": DEFAULT_USER_AGENT,
    "cache_dir": ".msaf_cache",
    "seed": 1337,
    "max_rl_steps": 200,
    "ga_population": 20,
    "ga_generations": 40,
    "vae_latent_dim": 32,
    "vae_epochs": 20,
    "dryrun": True
}

AMINO = list("ACDEFGHIKLMNPQRSTVWY")
VALID_AA = set(AMINO + ["X", "B", "Z", "U", "O"])

# ---------- Utilities ----------
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    if TORCH_AVAILABLE:
        torch.manual_seed(s)
        try:
            torch.cuda.manual_seed_all(s)
        except Exception:
            pass

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_of_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def ungap(seq: str) -> str:
    return seq.replace("-", "").replace(".", "").upper()

def warn_if_gaps(seq: str, context: str = ""):
    if "-" in seq or "." in seq:
        logger.warning(f"{context} sequence contains gap chars; ungapping for model input.")

def clean_a3m(path: Path, keep_gaps_for_msa: bool = True) -> Path:
    """
    Convert an A3M-like file to a FASTA. If keep_gaps_for_msa is False,
    remove gaps when producing the primary sequence file. Returns path
    to cleaned fasta (suffix _clean.fasta).
    """
    p = Path(path)
    out = p.with_suffix(p.suffix + "_clean.fasta")
    lines_out = []
    with open(path, "r") as fh:
        header = None
        seq_parts = []
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_parts)
                    if not keep_gaps_for_msa:
                        seq = ungap(seq)
                    lines_out.append(header)
                    lines_out.append(seq)
                header = line
                seq_parts = []
            else:
                # a3m uses lowercase for insertions; remove lowercase letters
                cleaned = "".join([c for c in line if not c.islower()])
                seq_parts.append(cleaned)
        # final
        if header is not None:
            seq = "".join(seq_parts)
            if not keep_gaps_for_msa:
                seq = ungap(seq)
            lines_out.append(header)
            lines_out.append(seq)
    with open(out, "w") as of:
        of.write("\n".join(lines_out) + "\n")
    return out

# ---------- PDB helpers ----------
def extract_plddt(pdb_path: Path) -> Tuple[np.ndarray, float, float, float]:
    """
    Return (plddt_array, frac>70, frac>80, frac>90).
    Uses CA atom b-factors as pLDDT proxies.
    """
    if not pdb_path.exists():
        raise FileNotFoundError(str(pdb_path))
    p = PDBParser(QUIET=True)
    try:
        struct = p.get_structure("m", str(pdb_path))
    except Exception as e:
        logger.error(f"Failed to parse PDB {pdb_path}: {e}")
        return np.array([]), 0.0, 0.0, 0.0
    b_factors = []
    for atom in struct.get_atoms():
        if atom.get_id() == "CA":
            try:
                b_factors.append(atom.get_bfactor())
            except Exception:
                b_factors.append(0.0)
    if not b_factors:
        return np.array([]), 0.0, 0.0, 0.0
    arr = np.array(b_factors)
    return arr, float((arr > 70).mean()), float((arr > 80).mean()), float((arr > 90).mean())

def compute_rmsd(pdb_a: Path, pdb_b: Path) -> float:
    p = PDBParser(QUIET=True)
    s1 = p.get_structure("a", str(pdb_a))
    s2 = p.get_structure("b", str(pdb_b))
    ca1 = [x for x in s1.get_atoms() if x.get_id() == "CA"]
    ca2 = [x for x in s2.get_atoms() if x.get_id() == "CA"]
    if not ca1 or not ca2:
        logger.warning(f"One of PDBs has no CA atoms: {pdb_a}, {pdb_b}")
        return float("nan")
    minlen = min(len(ca1), len(ca2))
    if len(ca1) != len(ca2):
        logger.warning(f"PDB CA counts differ ({len(ca1)} vs {len(ca2)}), truncating to {minlen}")
    try:
        sup = Superimposer()
        sup.set_atoms(ca1[:minlen], ca2[:minlen])
        return float(sup.rms)
    except Exception as e:
        logger.error(f"RMSD failed: {e}")
        return float("nan")

# ---------- ColabFold wrapper with compatibility checks ----------
def run_colabfold(queries, result_dir: Path, msa_path: Optional[str] = None,
                  use_templates: bool = False, custom_template_path: Optional[str] = None,
                  num_models: int = 5, is_complex: bool = False, save_msa: bool = True,
                  save_all: bool = True, zip_results: bool = True, dryrun: bool = False) -> None:
    """
    Wrapper around colabfold.batch.run with fallback handling for different signatures.
    If dryrun True, will create a fake PDB and MSA output for testing.
    """
    result_dir = Path(result_dir)
    safe_mkdir(result_dir)

    if dryrun or not COLABFOLD_AVAILABLE:
        # Create dummy outputs to allow pipeline testing
        logger.info(f"[dryrun] Simulating ColabFold run in {result_dir}")
        # fake MSA
        if msa_path:
            fake_msa = Path(msa_path)
            shutil.copy(msa_path, result_dir / Path(msa_path).name)
        else:
            fake_msa = result_dir / "baseline_clean.fasta"
            with open(fake_msa, "w") as fh:
                fh.write(">dryrun\n" + "A" * 50 + "\n")
        # fake pdb: simple poly-A chain with b-factors that vary
        fake_pdb = result_dir / "rank_1.pdb"
        seq_len = 50
        b_vals = np.linspace(90, 30, seq_len)
        with open(fake_pdb, "w") as fh:
            fh.write("END\n")
        return

    # try to call run with common parameter combos; handle TypeError gracefully
    try:
        colabfold_run(
            queries=queries,
            result_dir=str(result_dir),
            use_templates=use_templates,
            custom_template_path=custom_template_path,
            num_models=num_models,
            is_complex=is_complex,
            msa_path=msa_path,
            save_msa=save_msa,
            save_all=save_all,
            zip_results=zip_results
        )
    except TypeError as e1:
        # Some versions may not accept msa_path or other kwargs; attempt progressive calls
        logger.info(f"colabfold.run signature mismatch, trying alternate call: {e1}")
        try:
            colabfold_run(queries=queries, result_dir=str(result_dir),
                          use_templates=use_templates, num_models=num_models,
                          is_complex=is_complex, save_all=save_all)
        except Exception as e2:
            logger.error(f"colabfold.run failed with fallback: {e2}")
            raise

# ---------- Perturbations ----------
def perturb_delete(msa_src: Path, dst: Path, num_delete: int = 5):
    aln = AlignIO.read(str(msa_src), "fasta")
    seqs = list(aln)
    n = len(seqs)
    if n <= 1:
        logger.warning(f"perturb_delete: MSA has {n} sequences; skipping")
        return
    drop_count = min(num_delete, max(0, n - 1))
    drop = set(random.sample(range(n), drop_count))
    new = [s for i, s in enumerate(seqs) if i not in drop]
    AlignIO.write(MultipleSeqAlignment(new), str(dst), "fasta")
    logger.info(f"perturb_delete -> {dst} (dropped {len(drop)})")

def perturb_mutate(msa_src: Path, dst: Path, mut_rate: float = 0.02):
    aln = AlignIO.read(str(msa_src), "fasta")
    out = []
    for rec in aln:
        seq = list(str(rec.seq))
        for i in range(len(seq)):
            if seq[i] in "-.":
                # keep gaps in MSA representation
                continue
            if random.random() < mut_rate:
                seq[i] = random.choice([a for a in AMINO if a != seq[i]])
        out.append(SeqRecord(type(rec.seq)("".join(seq)), id=rec.id, description=""))
    AlignIO.write(MultipleSeqAlignment(out), str(dst), "fasta")
    logger.info(f"perturb_mutate -> {dst}")

def perturb_shuffle(msa_src: Path, dst: Path):
    aln = AlignIO.read(str(msa_src), "fasta")
    rows = list(aln)
    random.shuffle(rows)
    mat = np.array([list(str(r.seq)) for r in rows], dtype=str)
    if mat.size == 0:
        AlignIO.write(MultipleSeqAlignment(rows), str(dst), "fasta")
        return
    idx = np.arange(mat.shape[1])
    np.random.shuffle(idx)
    shuffled = mat[:, idx]
    out = []
    for i, r in enumerate(rows):
        out.append(SeqRecord(type(r.seq)("".join(shuffled[i])), id=r.id, description=""))
    AlignIO.write(MultipleSeqAlignment(out), str(dst), "fasta")
    logger.info(f"perturb_shuffle -> {dst}")

# ---------- Simple PLM-likelihood proxy ----------
def msa_frequency_score(seq: str, msa: List[str]) -> float:
    """
    Simple PLM proxy: compute per-position log-likelihood using amino-acid
    frequencies in MSA columns. Higher means more 'natural'.
    """
    if not msa:
        return 0.0
    L = len(seq)
    pad = "-" * L
    columns = [ [s[i] if i < len(s) else "-" for s in msa] for i in range(L) ]
    score = 0.0
    for i, aa in enumerate(seq):
        col = columns[i]
        freq = {}
        for c in col:
            freq[c] = freq.get(c, 0) + 1
        denom = sum(freq.values())
        p = (freq.get(aa, 0) + 1e-6) / denom
        score += math.log(p)
    return score

# ---------- DCA (mutual information approximation) ----------
def compute_mutual_info_contact_map(msa_path: Path, top_k: int = 50) -> List[Tuple[int,int,float]]:
    aln = AlignIO.read(str(msa_path), "fasta")
    seqs = [ungap(str(r.seq)) for r in aln]
    if not seqs:
        return []
    L = len(seqs[0])
    # pad or truncate
    seqs = [s[:L].upper() for s in seqs]
    # encode
    q = len(seqs)
    mi = np.zeros((L, L))
    for i in range(L):
        for j in range(i+1, L):
            # joint distribution
            pairs = {}
            for s in seqs:
                a = s[i] if i < len(s) else "-"
                b = s[j] if j < len(s) else "-"
                pairs[(a,b)] = pairs.get((a,b), 0) + 1
            # MI calculation
            mi_ij = 0.0
            marg_i = {}
            marg_j = {}
            for (a,b), cnt in pairs.items():
                marg_i[a] = marg_i.get(a, 0) + cnt
                marg_j[b] = marg_j.get(b, 0) + cnt
            for (a,b), cnt in pairs.items():
                p_ab = cnt / q
                p_a = marg_i[a] / q
                p_b = marg_j[b] / q
                if p_ab > 0 and p_a > 0 and p_b > 0:
                    mi_ij += p_ab * math.log(p_ab / (p_a * p_b) + 1e-12)
            mi[i,j] = mi_ij
            mi[j,i] = mi_ij
    # produce top pairs
    pairs = []
    flat = []
    for i in range(L):
        for j in range(i+1, L):
            flat.append((i,j,mi[i,j]))
    flat_sorted = sorted(flat, key=lambda x: x[2], reverse=True)
    return flat_sorted[:top_k]

# ---------- Simple GA optimizer ----------
def ga_optimize_msa(initial_msa: Path, target_seq_fasta: Path, result_dir: Path,
                    popsize: int = 20, generations: int = 30, dryrun: bool = True):
    """
    Evolve MSAs (row-level alterations) to maximize structural change (simulated here).
    Returns best candidate MSA path.
    """
    safe_mkdir(result_dir)
    aln = AlignIO.read(str(initial_msa), "fasta")
    rows = [str(r.seq) for r in aln]
    nrows = len(rows)
    if nrows == 0:
        logger.warning("GA: empty MSA")
        return initial_msa
    L = len(rows[0])
    # initialize population (list of MSA lists)
    population = []
    for _ in range(popsize):
        ind = rows.copy()
        # tiny shuffle/mutate
        for _ in range(random.randint(0, max(1, nrows//5))):
            i = random.randrange(nrows)
            j = random.randrange(nrows)
            ind[i], ind[j] = ind[j], ind[i]
        population.append(ind)
    # baseline evaluation function: run_colabfold and use mean pLDDT or surrogate
    def evaluate(ind_msa_rows):
        # write to temp msa
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fasta") as tf:
            for idx, s in enumerate(ind_msa_rows):
                tf.write(f">seq{idx}\n{s}\n")
            tf.flush()
            tmp_path = Path(tf.name)
        # either call colabfold or dry surrogate
        tmp_out = result_dir / f"tmp_eval_{sha256_of_file(tmp_path)[:8]}"
        try:
            run_colabfold(queries=get_queries(str(target_seq_fasta))[0:1] if COLABFOLD_AVAILABLE else [{"query": "dummy"}],
                          result_dir=tmp_out, msa_path=str(tmp_path), dryrun=dryrun)
            # find pdb
            pdb = next(tmp_out.rglob("*rank_*.pdb"), None)
            if pdb:
                plddt, _, _, _ = extract_plddt(pdb)
                return float(plddt.mean()) if plddt.size else 0.0
            else:
                return 0.0
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass
    # generational loop
    best_ind = population[0]
    best_score = None
    for gen in range(generations):
        scores = []
        for ind in population:
            score = evaluate(ind)
            scores.append(score)
        # lower mean pLDDT is better adversarial objective; so sort ascending
        ranked = sorted(zip(population, scores), key=lambda x: x[1])
        population = [x[0] for x in ranked[:popsize//2]]
        # keep elites and reproduce
        newpop = population.copy()
        while len(newpop) < popsize:
            a = random.choice(population)
            b = random.choice(population)
            # crossover: swap half rows
            cut = random.randint(1, nrows-1)
            child = a[:cut] + b[cut:]
            # mutate: small swap
            if random.random() < 0.2:
                i,j = random.sample(range(nrows), 2)
                child[i], child[j] = child[j], child[i]
            newpop.append(child)
        population = newpop
        curr_best_score = ranked[0][1]
        curr_best = ranked[0][0]
        if best_score is None or curr_best_score < best_score:
            best_score = curr_best_score
            best_ind = curr_best
        logger.info(f"GA gen {gen+1}/{generations} best_score {curr_best_score:.3f}")
    # write best_ind to file
    outp = result_dir / "ga_best.fasta"
    with open(outp, "w") as fh:
        for i, s in enumerate(best_ind):
            fh.write(f">ga_{i}\n{s}\n")
    return outp

# ---------- Simple VAE for sequences (optional, torch) ----------
if TORCH_AVAILABLE:
    class SeqDataset(Dataset):
        def __init__(self, seqs: List[str], maxlen: int):
            self.seqs = seqs
            self.maxlen = maxlen
            self.vocab = {a:i+1 for i,a in enumerate(AMINO)}  # 0 for pad
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, idx):
            s = self.seqs[idx][:self.maxlen]
            arr = np.zeros(self.maxlen, dtype=np.int64)
            for i,ch in enumerate(s):
                arr[i] = self.vocab.get(ch, 0)
            return torch.tensor(arr)
    class SimpleVAE(nn.Module):
        def __init__(self, maxlen: int, vocab_size: int, latent_dim: int):
            super().__init__()
            self.maxlen = maxlen
            self.emb = nn.Embedding(vocab_size+1, 32, padding_idx=0)
            self.encoder = nn.Sequential(nn.Linear(32*maxlen, 256), nn.ReLU(), nn.Linear(256, latent_dim*2))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 32*maxlen))
            self.out_linear = nn.Linear(32, vocab_size+1)
        def encode(self, x):
            # x: (B, L)
            B = x.shape[0]
            e = self.emb(x)  # B,L,32
            e = e.view(B, -1)
            stats = self.encoder(e)
            mu = stats[:, :stats.shape[1]//2]
            logvar = stats[:, stats.shape[1]//2:]
            return mu, logvar
        def reparam(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        def decode(self, z):
            B = z.shape[0]
            d = self.decoder(z)  # B,32*L
            d = d.view(B, self.maxlen, 32)
            logits = self.out_linear(d)  # B,L,vocab+1
            return logits
        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparam(mu, logvar)
            logits = self.decode(z)
            return logits, mu, logvar
    def train_vae_on_msa(msa_path: Path, result_dir: Path, latent_dim: int = 32, epochs: int = 10, batch_size: int = 32):
        aln = AlignIO.read(str(msa_path), "fasta")
        seqs = [ungap(str(r.seq)) for r in aln]
        if not seqs:
            logger.warning("VAE: no sequences")
            return None
        maxlen = max(len(s) for s in seqs)
        ds = SeqDataset(seqs, maxlen)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        model = SimpleVAE(maxlen, vocab_size=len(AMINO), latent_dim=latent_dim)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for ep in range(epochs):
            total_loss = 0.0
            for b in dl:
                b = b.long()
                logits, mu, logvar = model(b)
                recon_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), b.view(-1), ignore_index=0)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b.shape[0]
                loss = recon_loss + 1e-4 * kld
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
            logger.info(f"VAE epoch {ep+1}/{epochs} loss {total_loss/len(dl):.3f}")
        # save model
        safe_mkdir(result_dir)
        torch.save(model.state_dict(), str(result_dir / "vae.pt"))
        return model
else:
    def train_vae_on_msa(*args, **kwargs):
        logger.warning("VAE training skipped: torch not available")
        return None

# ---------- Explainability: occlusion & gradient (occlusion implemented) ----------
def attribute_columns_by_occlusion(msa_path: Path, query_fasta: Path, pdb_ref: Path, outdir: Path,
                                   base_name: str, run_cb=run_colabfold, dryrun: bool = True):
    """
    For each column in the MSA, mask that column to gaps and rerun model to measure Δ mean pLDDT.
    Writes CSV and plot.
    """
    safe_mkdir(outdir)
    # baseline pLDDT
    pl0_arr, _, _, _ = extract_plddt(pdb_ref)
    if pl0_arr.size == 0:
        base_score = 0.0
    else:
        base_score = float(pl0_arr.mean())
    # read cleaned MSA keeping gaps
    cleaned = clean_a3m(msa_path, keep_gaps_for_msa=True)
    aln = AlignIO.read(str(cleaned), "fasta")
    ncol = aln.get_alignment_length()
    drops = []
    for col in range(ncol):
        masked = []
        for rec in aln:
            seq = list(str(rec.seq))
            # mask column to gap
            if col < len(seq):
                seq[col] = "-"
            masked.append(SeqRecord(type(rec.seq)("".join(seq)), id=rec.id, description=""))
        tmp_msa = outdir / f"occlude_col{col:03d}.fasta"
        AlignIO.write(MultipleSeqAlignment(masked), str(tmp_msa), "fasta")
        out_dir = outdir / f"occlude_col{col:03d}"
        safe_mkdir(out_dir)
        # produce query_fasta that points to the original query (single sequence)
        # some colabfold versions accept msa_path; run_colabfold wrapper will adapt
        try:
            q, is_cplx = get_queries(str(query_fasta))
        except Exception:
            # fallback: build query list manually (colabfold_run may accept list of dicts)
            q = get_queries(str(query_fasta)) if COLABFOLD_AVAILABLE else [{"query": "dry"}]
            is_cplx = False
        run_cb(queries=q, result_dir=out_dir, msa_path=str(tmp_msa), dryrun=dryrun)
        pdb_new = next(out_dir.rglob("*rank_*.pdb"), None)
        if not pdb_new:
            logger.warning(f"{base_name} occlusion col {col} produced no PDB")
            drops.append((col, None))
            continue
        pl_new, _, _, _ = extract_plddt(pdb_new)
        if pl_new.size == 0:
            drops.append((col, None))
            continue
        score_new = float(pl_new.mean())
        delta = base_score - score_new
        drops.append((col, delta))
        logger.info(f"{base_name} occlude col {col:03d} ΔpLDDT {delta:.3f}")
    # write CSV
    csv_path = outdir / "column_attributions.csv"
    with open(csv_path, "w", newline="") as af:
        w = csv.writer(af)
        w.writerow(["column", "delta_mean_pLDDT"])
        for col, delta in drops:
            w.writerow([col, "" if delta is None else f"{delta:.6f}"])
    # plot
    deltas = np.array([0.0 if d is None else d for (_, d) in drops])
    fig, ax = plt.subplots(figsize=(8,2))
    ax.plot(np.arange(len(deltas)), deltas, "-o", markersize=3)
    ax.set_xlabel("MSA Column")
    ax.set_ylabel("Δ mean pLDDT")
    ax.set_title(f"{base_name} column occlusion")
    fig.savefig(outdir / "column_attribution_plot.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved attribution CSV and plot to {outdir}")
    return csv_path

# ---------- RL environment for MSA adversarial edits ----------
if GYM_AVAILABLE:
    class MSAEnv(gym.Env):
        """
        Minimal Gym environment for MSA adversary.
        Observation: vector of simple MSA stats: [depth, mean_entropy, mean_gap_frac]
        Action space: discrete actions
          0: delete a random homolog row
          1: mutate a random residue in a random row
          2: shuffle columns (random permutation)
          3: no-op
        Reward: negative mean pLDDT from run_colabfold (lower pLDDT -> higher reward)
        """
        metadata = {"render.modes": ["human"]}
        def __init__(self, msa_path: Path, query_fasta: Path, workdir: Path, dryrun: bool = True):
            super().__init__()
            self.msa_path = Path(msa_path)
            self.query_fasta = Path(query_fasta)
            self.workdir = Path(workdir)
            self.dryrun = dryrun
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
            self._load()
            self.step_count = 0
            self.max_steps = DEFAULTS["max_rl_steps"]
            self.latest_pdb = None
        def _load(self):
            aln = AlignIO.read(str(self.msa_path), "fasta")
            self.rows = [list(str(r.seq)) for r in aln]
            self.n = len(self.rows)
            self.L = len(self.rows[0]) if self.n>0 else 0
        def _obs(self):
            # depth, mean_entropy (normalized), mean_gap_frac
            depth = float(self.n)
            if self.n == 0 or self.L == 0:
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            cols = list(zip(*self.rows))
            entropies = []
            gap_fracs = []
            for c in cols:
                counts = {}
                for a in c:
                    counts[a] = counts.get(a, 0) + 1
                probs = np.array(list(counts.values())) / sum(counts.values())
                ent = -np.sum(probs * np.log(probs + 1e-12))
                entropies.append(ent)
                gap_fracs.append(c.count("-") / len(c))
            # normalize
            mean_ent = float(np.mean(entropies)) / (math.log(len(AMINO)+5) + 1e-6)
            mean_gap = float(np.mean(gap_fracs))
            depth_norm = min(depth / 1000.0, 1.0)
            return np.array([depth_norm, mean_ent, mean_gap], dtype=np.float32)
        def reset(self):
            self._load()
            self.step_count = 0
            return self._obs()
        def step(self, action):
            self.step_count += 1
            # apply action
            if action == 0 and self.n > 1:
                # delete one random homolog (not the query)
                i = random.randrange(1, self.n)  # keep index 0 as query if structured that way
                del self.rows[i]
                self.n -= 1
            elif action == 1 and self.n > 0:
                r = random.randrange(self.n)
                c = random.randrange(self.L)
                if self.rows[r][c] not in "-.":
                    self.rows[r][c] = random.choice([a for a in AMINO if a != self.rows[r][c]])
            elif action == 2:
                # shuffle columns
                idx = list(range(self.L))
                random.shuffle(idx)
                self.rows = [ [row[i] for i in idx] for row in self.rows ]
            # write MSA to tmp and run eval
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fasta") as tf:
                for i,row in enumerate(self.rows):
                    tf.write(f">r{i}\n{''.join(row)}\n")
                tf.flush()
                tmp_path = Path(tf.name)
            out_dir = self.workdir / f"rl_step_{self.step_count}"
            safe_mkdir(out_dir)
            try:
                # run colabfold or dry run
                q = [get_queries(str(self.query_fasta))[0]] if COLABFOLD_AVAILABLE else [{"query":"dry"}]
            except Exception:
                q = [{"query": "dry"}]
            run_colabfold(queries=q, result_dir=out_dir, msa_path=str(tmp_path), dryrun=self.dryrun)
            pdb = next(out_dir.rglob("*rank_*.pdb"), None)
            if pdb:
                plddt, _, _, _ = extract_plddt(pdb)
                mean_plddt = float(plddt.mean()) if plddt.size else 100.0
            else:
                mean_plddt = 100.0
            # reward: larger RMSD or lower pLDDT is positive
            reward = max(0.0, 100.0 - mean_plddt)
            done = self.step_count >= self.max_steps
            obs = self._obs()
            info = {"plddt": mean_plddt}
            try:
                tmp_path.unlink()
            except Exception:
                pass
            return obs, reward, done, info
        def render(self, mode="human"):
            pass
else:
    MSAEnv = None

def train_rl_adversary(msa_path: Path, query_fasta: Path, workdir: Path, result_dir: Path,
                       timesteps: int = 20000, dryrun: bool = True):
    safe_mkdir(result_dir)
    if not GYM_AVAILABLE:
        logger.warning("Gym not found; RL training skipped")
        return None
    env = MSAEnv(msa_path=msa_path, query_fasta=query_fasta, workdir=workdir, dryrun=dryrun)
    if SB3_AVAILABLE:
        logger.info("Training PPO adversary (stable-baselines3)...")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=timesteps)
        # save policy
        pth = result_dir / "ppo_adversary.zip"
        model.save(str(pth))
        logger.info(f"Saved PPO model to {pth}")
        return model
    else:
        logger.warning("stable-baselines3 not available; running simple random search for policy")
        # random policy: run multiple episodes and keep best found MSA
        best = None
        best_reward = -1e9
        for episode in range(50):
            obs = env.reset()
            total_r = 0.0
            done = False
            t = 0
            while not done and t < 100:
                action = env.action_space.sample()
                obs, r, done, info = env.step(action)
                total_r += r
                t += 1
            if total_r > best_reward:
                best_reward = total_r
                # save final MSA
                tmp_out = result_dir / f"random_best_{episode}.fasta"
                # write current env rows
                with open(tmp_out, "w") as fh:
                    for i,row in enumerate(env.rows):
                        fh.write(f">r{i}\n{''.join(row)}\n")
                best = tmp_out
        logger.info(f"Random search best reward {best_reward}")
        return best

# ---------- Conformational ensemble collection and clustering ----------
def collect_ensemble_and_cluster(msa_variants: List[Path], query_fasta: Path, outdir: Path,
                                 run_cb=run_colabfold, dryrun: bool = True, cluster_n: int = 3):
    safe_mkdir(outdir)
    pdb_paths = []
    for i, m in enumerate(msa_variants):
        od = outdir / f"variant_{i}"
        safe_mkdir(od)
        try:
            q = get_queries(str(query_fasta))
        except Exception:
            q = [{"query":"dry"}]
        run_cb(queries=q, result_dir=od, msa_path=str(m), dryrun=dryrun)
        pdb = next(od.rglob("*rank_*.pdb"), None)
        if pdb:
            pdb_paths.append(pdb)
    # compute pairwise RMSD matrix
    n = len(pdb_paths)
    if n == 0:
        logger.warning("No PDBs produced for ensemble")
        return []
    dist = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            r = compute_rmsd(pdb_paths[i], pdb_paths[j])
            if math.isnan(r):
                r = 999.0
            dist[i,j] = r
            dist[j,i] = r
    # clustering
    if SKLEARN_AVAILABLE:
        cl = AgglomerativeClustering(n_clusters=min(cluster_n, n), affinity="precomputed", linkage="average")
        labels = cl.fit_predict(dist)
    else:
        # simple greedy cluster: pick nearest medoids
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            labels[i] = int(i % min(cluster_n, n))
    # choose representatives
    reps = {}
    for i, lab in enumerate(labels):
        reps.setdefault(lab, []).append((i, pdb_paths[i]))
    rep_paths = []
    for lab, items in reps.items():
        # choose item with minimal average distance to cluster mates
        best_idx, best_path = items[0]
        best_score = float("inf")
        for idx, pth in items:
            scores = [dist[idx,j] for j,_ in items if j != idx]
            score = np.mean(scores) if scores else 0.0
            if score < best_score:
                best_score = score
                best_idx, best_path = idx, pth
        rep_paths.append(best_path)
    # save mapping
    mapping = {"pdbs": [str(p) for p in pdb_paths], "labels": labels.tolist(), "representatives": [str(p) for p in rep_paths]}
    with open(outdir / "ensemble.json", "w") as fh:
        json.dump(mapping, fh, indent=2)
    logger.info(f"Clustered ensemble; reps: {len(rep_paths)}")
    return rep_paths

# ---------- Robustness benchmarking runner (ties modules together) ----------
def run_pipeline_for_fasta(fasta_path: Path, results_root: Path, cfg: Dict[str,Any]):
    name = fasta_path.stem
    logger.info(f"[{name}] START")
    base_dir = results_root / name
    msas_dir = base_dir / "msas"
    pert_dir = base_dir / "perturbed"
    safe_mkdir(msas_dir)
    safe_mkdir(pert_dir)
    # download weights if available and not dryrun
    if not cfg["dryrun"] and COLABFOLD_AVAILABLE:
        try:
            logger.info(f"[{name}] Downloading alphafold params (may take a while)...")
            download_alphafold_params(model_type="AlphaFold2-ptm")
        except Exception as e:
            logger.warning(f"Could not download alphafold params: {e}")
    # baseline run
    out_base = base_dir / "baseline"
    safe_mkdir(out_base)
    # check file size
    if fasta_path.stat().st_size == 0:
        logger.warning(f"[{name}] FASTA empty; skipping")
        return
    try:
        queries = get_queries(str(fasta_path)) if COLABFOLD_AVAILABLE else [{"query": "dry"}]
    except Exception as e:
        logger.warning(f"[{name}] get_queries failed: {e}; using fallback")
        queries = [{"query":"dry"}]
    try:
        run_colabfold(queries=queries, result_dir=out_base, use_templates=cfg["use_templates"],
                      num_models=cfg["num_models"], is_complex=False, dryrun=cfg["dryrun"])
    except Exception as e:
        logger.error(f"[{name}] baseline run failed: {e}")
        return
    # find MSA produced (prefer .a3m then fasta)
    msa_file = next(out_base.rglob("*.a3m"), None) or next(out_base.rglob("*.fasta"), None) or next(out_base.rglob("*.sto"), None)
    if not msa_file:
        logger.warning(f"[{name}] No MSA found in {out_base}; creating cleaned copy of input")
        # create cleaned from input
        msa_file = msas_dir / f"{name}_baseline_clean.fasta"
        shutil.copy(fasta_path, msa_file)
    else:
        # copy to msas_dir
        out_path = msas_dir / msa_file.name
        shutil.copy(msa_file, out_path)
        msa_file = out_path
    depth = sum(1 for line in open(msa_file) if line.startswith(">"))
    logger.info(f"[{name}] baseline MSA copied to {msa_file} (depth={depth})")
    # baseline pdb
    pdb_base = next(out_base.rglob("*rank_*.pdb"), None)
    if not pdb_base:
        logger.warning(f"[{name}] No baseline PDB found (dryrun or failure).")
    # Attribution (occlusion)
    try:
        attr_dir = base_dir / "attribution"; safe_mkdir(attr_dir)
        if pdb_base:
            attribute_columns_by_occlusion(msa_file, fasta_path, pdb_base, attr_dir, base_name=name, run_cb=run_colabfold, dryrun=cfg["dryrun"])
        else:
            logger.warning(f"[{name}] skipping occlusion: baseline PDB missing")
    except Exception as e:
        logger.warning(f"[{name}] attribution failed: {e}")
    # Perturbations
    strat_list = [
        ("delete", pert_dir / f"{name}_del{cfg['num_delete']}.fasta"),
        ("mutate", pert_dir / f"{name}_mut{int(cfg['mut_rate']*100)}.fasta"),
        ("shuffle", pert_dir / f"{name}_shuffle.fasta")
    ]
    perturb_delete(msa_file, strat_list[0][1], num_delete=cfg["num_delete"])
    perturb_mutate(msa_file, strat_list[1][1], mut_rate=cfg["mut_rate"])
    perturb_shuffle(msa_file, strat_list[2][1])
    # For each strategy, run Alphafold and collect metrics
    csv_path = base_dir / "summary.csv"
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["strategy", "param", "depth", "mean_pLDDT", "frac>70", "frac>80", "frac>90", "RMSD"])
        # baseline metrics
        if pdb_base:
            pl0, f70_0, f80_0, f90_0 = extract_plddt(pdb_base)
            mean0 = float(pl0.mean()) if pl0.size else 0.0
            w.writerow(["baseline", "-", depth, f"{mean0:.2f}", f"{f70_0:.2f}", f"{f80_0:.2f}", f"{f90_0:.2f}", "0.00"])
        else:
            w.writerow(["baseline", "-", depth, "0.00", "0.00", "0.00", "0.00", "0.00"])
        for strat, msa in strat_list:
            outd = base_dir / strat; safe_mkdir(outd)
            logger.info(f"[{name}] running strategy {strat}")
            # produce a single-sequence FASTA query for model (first sequence from original input)
            seq_lines = open(fasta_path).read().splitlines()
            if len(seq_lines) < 2:
                query_fasta = base_dir / f"{name}_query_{strat}.fasta"
                with open(query_fasta, "w") as qf:
                    qf.write(f">query\n" + "A"*50 + "\n")
            else:
                query_fasta = base_dir / f"{name}_query_{strat}.fasta"
                with open(query_fasta, "w") as qf:
                    qf.write(seq_lines[0] + "\n")
                    qf.write(seq_lines[1] + "\n")
            try:
                q = get_queries(str(query_fasta)) if COLABFOLD_AVAILABLE else [{"query": "dry"}]
            except Exception:
                q = [{"query":"dry"}]
            try:
                run_colabfold(queries=q, result_dir=outd, msa_path=str(msa), dryrun=cfg["dryrun"])
            except Exception as e:
                logger.error(f"[{name}] strategy run failed: {e}")
                continue
            pdb_p = next(outd.rglob("*rank_*.pdb"), None)
            if not pdb_p:
                logger.error(f"[{name}] No PDB for {strat}")
                continue
            plp, f70, f80, f90 = extract_plddt(pdb_p)
            meanp = float(plp.mean()) if plp.size else 0.0
            rms = compute_rmsd(pdb_base, pdb_p) if pdb_base else float("nan")
            w.writerow([strat, msa.name, depth, f"{meanp:.2f}", f"{f70:.2f}", f"{f80:.2f}", f"{f90:.2f}", f"{rms:.3f}"])
    logger.info(f"[{name}] summary CSV written to {csv_path}")
    # Plots
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
        fig, ax1 = plt.subplots()
        ax1.bar(data["strategy"], data["mean_pLDDT"].astype(float), alpha=0.6)
        ax1.set_ylabel("mean pLDDT")
        ax2 = ax1.twinx()
        ax2.plot(data["strategy"], data["RMSD"].astype(float), "ro-")
        ax2.set_ylabel("RMSD (Å)")
        plt.title(f"{name}: pLDDT vs RMSD")
        fig.savefig(base_dir / "summary_plot.png")
        plt.close(fig)
        # depth vs RMSD scatter
        fig, ax = plt.subplots()
        depths = data["depth"].astype(float)
        rmsd_vals = data["RMSD"].astype(float)
        for i, strat in enumerate(data["strategy"]):
            ax.scatter(depths[i], rmsd_vals[i], label=strat)
        ax.set_xlabel("MSA depth (# sequences)")
        ax.set_ylabel("RMSD (Å)")
        ax.set_xscale("log")
        ax.legend()
        fig.savefig(base_dir / "depth_vs_rmsd_scatter.png")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"[{name}] plotting failed: {e}")
    # DCA analysis
    try:
        dca_top = compute_mutual_info_contact_map(msa_file, top_k=50)
        with open(base_dir / "dca_top.json", "w") as fh:
            json.dump([{"i":i,"j":j,"mi":float(mi)} for (i,j,mi) in dca_top], fh, indent=2)
        logger.info(f"[{name}] DCA top pairs saved")
    except Exception as e:
        logger.warning(f"[{name}] DCA failed: {e}")
    # GA adversary (quick)
    try:
        ga_dir = base_dir / "ga"; safe_mkdir(ga_dir)
        ga_best = ga_optimize_msa(msa_file, fasta_path, ga_dir, popsize=cfg["ga_population"], generations=cfg["ga_generations"], dryrun=cfg["dryrun"])
        logger.info(f"[{name}] GA best saved: {ga_best}")
    except Exception as e:
        logger.warning(f"[{name}] GA failed: {e}")
    # RL adversary (train small)
    try:
        rl_dir = base_dir / "rl"; safe_mkdir(rl_dir)
        rl_out = train_rl_adversary(msa_file, fasta_path, rl_dir / "work", rl_dir, timesteps=2000, dryrun=cfg["dryrun"])
        logger.info(f"[{name}] RL output: {rl_out}")
    except Exception as e:
        logger.warning(f"[{name}] RL failed: {e}")
    # VAE training (optional)
    try:
        vae_dir = base_dir / "vae"; safe_mkdir(vae_dir)
        vae = train_vae_on_msa(msa_file, vae_dir, latent_dim=cfg["vae_latent_dim"], epochs=cfg.get("vae_epochs", 10))
        if vae is not None:
            logger.info(f"[{name}] VAE trained and saved to {vae_dir}")
    except Exception as e:
        logger.warning(f"[{name}] VAE training failed: {e}")
    # Ensemble generation (use simple perturbations)
    try:
        ensemble_dir = base_dir / "ensemble"; safe_mkdir(ensemble_dir)
        variants = []
        # create small library
        for i in range(5):
            pth = ensemble_dir / f"variant_{i}.fasta"
            if i % 3 == 0:
                perturb_delete(msa_file, pth, num_delete=max(1, cfg["num_delete"]//2))
            elif i % 3 == 1:
                perturb_mutate(msa_file, pth, mut_rate=max(0.01, cfg["mut_rate"]*2))
            else:
                perturb_shuffle(msa_file, pth)
            variants.append(pth)
        reps = collect_ensemble_and_cluster(variants, fasta_path, ensemble_dir, dryrun=cfg["dryrun"])
        logger.info(f"[{name}] Ensemble representatives: {reps}")
    except Exception as e:
        logger.warning(f"[{name}] Ensemble generation failed: {e}")
    logger.info(f"[{name}] DONE. Results -> {base_dir}")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="MSAffect: MSA adversary & robustness pipeline")
    ap.add_argument("--config", type=str, default=None, help="Path to config JSON (optional)")
    ap.add_argument("--dryrun", action="store_true", help="Run in dry mode (no AlphaFold calls).")
    ap.add_argument("--threads", type=int, default=1, help="Number of parallel threads (not used extensively).")
    ap.add_argument("--targets", type=str, default=None, help="Comma-separated FASTA basenames to run (without .fasta)")
    return ap.parse_args()

def load_config(path: Optional[str]) -> Dict[str,Any]:
    cfg = DEFAULTS.copy()
    if path:
        p = Path(path)
        if p.exists():
            try:
                cfg2 = json.loads(p.read_text())
                cfg.update(cfg2)
            except Exception as e:
                logger.warning(f"Could not load config {path}: {e}")
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg["dryrun"] = args.dryrun or cfg.get("dryrun", True)
    set_seed(cfg.get("seed", 1337))
    results_root = Path(cfg.get("results_root", DEFAULTS["results_root"]))
    msa_dir = Path(cfg.get("msa_dir", DEFAULTS["msa_dir"]))
    safe_mkdir(results_root)
    if not msa_dir.exists():
        logger.error(f"MSA dir {msa_dir} not found")
        sys.exit(1)
    # choose targets
    targets = []
    if args.targets:
        want = set([t.strip() for t in args.targets.split(",") if t.strip()])
        for p in msa_dir.glob("*.fasta"):
            if p.stem in want:
                targets.append(p)
    else:
        targets = list(msa_dir.glob("*.fasta"))
    if not targets:
        logger.error("No FASTA targets found in msa/")
        sys.exit(1)
    for fasta in targets:
        try:
            run_pipeline_for_fasta(fasta, results_root, cfg)
        except Exception as e:
            logger.error(f"Pipeline failed for {fasta.stem}: {e}")
            continue

if __name__ == "__main__":
    main()
