import os
import logging
import random
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, Superimposer

# ColabFold imports
from colabfold.batch import run, get_queries
from colabfold.download import download_alphafold_params

# ==== Logging Setup ====
def setup_logging():
    fmt = "%(levelname_color)s [%(asctime)s] %(message)s"
    datefmt = "%H:%M:%S"
    class ColorFormatter(logging.Formatter):
        COLORS = {
            "INFO": "\033[94m",    # blue
            "WARNING": "\033[93m", # yellow
            "ERROR": "\033[91m",   # red
        }
        def format(self, record):
            lvl = record.levelname
            color = self.COLORS.get(lvl, "")
            record.levelname_color = f"{color}{lvl}\033[0m"
            record.asctime = datetime.fromtimestamp(record.created).strftime("[%H:%M:%S]")
            return f"{record.levelname_color} {record.asctime} {record.getMessage()}"
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter(fmt, datefmt=datefmt))
    logging.root.handlers[:] = [handler]
    logging.root.setLevel(logging.INFO)

setup_logging()
logger = logging.getLogger(__name__)

# ==== Paths & Parameters ====
MSA_DIR         = Path("msas")
PERTURB_DIR     = Path("perturbed_msas")
RESULTS_DIR     = Path("results")
FASTA_DIR       = Path("msa")             # contains .fasta files
NUM_MODELS      = 5
USE_TEMPLATES   = False
MUTATION_RATE   = 0.02
NUM_DELETE      = 5

# Ensure dirs exist
for d in (MSA_DIR, PERTURB_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)

# ==== Step 1: Baseline Prediction & MSA Extraction ====
def baseline_run(fasta_path: Path):
    name = fasta_path.stem
    out_dir = RESULTS_DIR / name / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading params for {name}...")
    download_alphafold_params(model_type="AlphaFold2-ptm")
    logger.info(f"Running baseline prediction for {name}...")
    queries, is_complex = get_queries(str(fasta_path))
    run(queries=queries,
        result_dir=out_dir,
        use_templates=USE_TEMPLATES,
        custom_template_path=None,
        num_models=NUM_MODELS,
        is_complex=is_complex)
    # find and copy MSA
    msa_file = next(out_dir.glob("*.a3m"), None)
    if not msa_file:
        logger.error(f"No MSA found for {name}")
        return None, None
    msa_copy = MSA_DIR / f"{name}.a3m"
    msa_copy.write_bytes(msa_file.read_bytes())
    logger.info(f"Saved baseline MSA to {msa_copy}")
    pdb_file = next(out_dir.glob("ranked_*.pdb"), None)
    return msa_copy, pdb_file

# ==== Step 2: MSA Perturbations ====
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def random_delete(msa_in: Path, msa_out: Path, num_delete=NUM_DELETE):
    alignment = AlignIO.read(str(msa_in), "fasta")
    seqs = list(alignment)
    to_del = random.sample(range(len(seqs)), min(num_delete, len(seqs)-1))
    new = [s for i,s in enumerate(seqs) if i not in to_del]
    AlignIO.write(MultipleSeqAlignment(new), str(msa_out), "fasta")
    logger.info(f"Deleted {len(to_del)} seqs → {msa_out}")

def random_mutate(msa_in: Path, msa_out: Path, rate=MUTATION_RATE):
    alignment = AlignIO.read(str(msa_in), "fasta")
    mutated = []
    for rec in alignment:
        seq = list(str(rec.seq))
        for i in range(len(seq)):
            if random.random() < rate:
                seq[i] = random.choice([aa for aa in AMINO_ACIDS if aa!=seq[i]])
        mutated.append(SeqRecord(rec.seq.__class__("".join(seq)),
                                 id=rec.id, description=""))
    AlignIO.write(MultipleSeqAlignment(mutated), str(msa_out), "fasta")
    logger.info(f"Mutated msa at rate {rate} → {msa_out}")

# ==== Step 3 & 4: Rerun & Collect Metrics ====
def extract_plddt(pdb_path: Path):
    struct = PDBParser(QUIET=True).get_structure("m", str(pdb_path))
    return np.array([atm.get_bfactor() for atm in struct.get_atoms() if atm.get_id()=="CA"])

def compute_rmsd(pdb_a: Path, pdb_b: Path):
    p = PDBParser(QUIET=True)
    s1 = p.get_structure("a", str(pdb_a))
    s2 = p.get_structure("b", str(pdb_b))
    atoms1 = [a for a in s1.get_atoms() if a.get_id()=="CA"]
    atoms2 = [a for a in s2.get_atoms() if a.get_id()=="CA"]
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    return sup.rms

# ==== Full Pipeline ====
for fasta in FASTA_DIR.glob("*.fasta"):
    protein = fasta.stem
    # Baseline
    msa_base, pdb_base = baseline_run(fasta)
    if not msa_base:
        continue

    # Perturbations
    strategies = []
    # random_delete
    dd = PERTURB_DIR / f"{protein}_del{NUM_DELETE}.a3m"
    random_delete(msa_base, dd)
    strategies.append(("delete", dd))
    # random_mutate
    mm = PERTURB_DIR / f"{protein}_mut{int(MUTATION_RATE*100)}.a3m"
    random_mutate(msa_base, mm)
    strategies.append(("mutate", mm))

    # Prepare summary CSV
    summary_file = RESULTS_DIR / protein / "summary.csv"
    with open(summary_file, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["strategy","param","mean_pLDDT","RMSD"])

        # Baseline metrics
        base_pl = extract_plddt(pdb_base)
        writer.writerow(["baseline","-", f"{base_pl.mean():.2f}", "0.00"])

        # Rerun perturbed
        for strat, msa in strategies:
            outp = RESULTS_DIR / protein / f"{strat}"
            outp.mkdir(parents=True, exist_ok=True)
            logger.info(f"Rerunning {protein} with {strat} MSA...")
            # run with custom_msa_path
            queries, is_complex = get_queries(str(fasta))
            run(queries=queries,
                result_dir=outp,
                use_templates=USE_TEMPLATES,
                custom_msa_path=str(msa),
                custom_template_path=None,
                num_models=NUM_MODELS,
                is_complex=is_complex)
            pdb_pert = next(outp.glob("ranked_*.pdb"), None)
            pl = extract_plddt(pdb_pert)
            rmsd = compute_rmsd(pdb_base, pdb_pert)
            writer.writerow([strat, msa.name, f"{pl.mean():.2f}", f"{rmsd:.3f}"])

    logger.info(f"Summary saved to {summary_file}")

    # ==== Plot summary ====
    data = np.genfromtxt(summary_file, delimiter=",", dtype=None, names=True, encoding=None)
    fig, ax1 = plt.subplots()
    ax1.bar(data["strategy"], data["mean_pLDDT"].astype(float), alpha=0.6, label="mean pLDDT")
    ax1.set_ylabel("mean pLDDT")
    ax2 = ax1.twinx()
    ax2.plot(data["strategy"], data["RMSD"].astype(float), color="red", marker="o", label="RMSD")
    ax2.set_ylabel("RMSD (Å)")
    ax1.set_title(f"{protein}: pLDDT vs RMSD")
    fig.legend(loc="upper right")
    plt.savefig(RESULTS_DIR / protein / "summary_plot.png")
    plt.close()

    logger.info(f"Pipeline complete for {protein}. Summary plot saved.")
