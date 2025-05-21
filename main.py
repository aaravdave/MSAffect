
import os, random, csv, logging
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, Superimposer

from colabfold.batch import run, get_queries
from colabfold.download import download_alphafold_params

os.environ['HTTP_USER_AGENT'] = "MSAffect/0.1 contact@aaravdave.org"

# ─── Logging Setup ─────────────────────────────────────────────────────────────
def setup_logging():
    fmt = "%(levelname_color)s [%(asctime)s] %(message)s"
    datefmt = "%H:%M:%S"
    class ColorFormatter(logging.Formatter):
        COLORS = {"INFO":"\033[94m","WARNING":"\033[93m","ERROR":"\033[91m"}
        def format(self, record):
            lvl = record.levelname
            record.levelname_color = f"{self.COLORS.get(lvl,'')}{lvl}\033[0m"
            record.asctime = datetime.fromtimestamp(record.created).strftime("[%H:%M:%S]")
            return f"{record.levelname_color} {record.asctime} {record.getMessage()}"
    h = logging.StreamHandler()
    h.setFormatter(ColorFormatter(fmt, datefmt=datefmt))
    logging.root.handlers[:] = [h]
    logging.root.setLevel(logging.INFO)

setup_logging()
logger = logging.getLogger()

# ─── Paths & Parameters ───────────────────────────────────────────────────────
FASTA_DIR     = Path("msa")
RESULTS_ROOT  = Path("results")
NUM_MODELS    = 5
USE_TEMPLATES = False
MUT_RATE      = 0.02
NUM_DELETE    = 5

RESULTS_ROOT.mkdir(exist_ok=True)

# ─── Helpers ───────────────────────────────────────────────────────────────────
def extract_plddt(pdb_path):
    struct = PDBParser(QUIET=True).get_structure("m", str(pdb_path))
    b_factors = np.array([a.get_bfactor() for a in struct.get_atoms() if a.get_id()=="CA"])
    return b_factors, (b_factors > 70).mean(), (b_factors > 80).mean(), (b_factors > 90).mean()

def compute_rmsd(a, b):
    p = PDBParser(QUIET=True)
    s1, s2 = p.get_structure("a", str(a)), p.get_structure("b", str(b))
    ca1 = [x for x in s1.get_atoms() if x.get_id() == "CA"]
    ca2 = [x for x in s2.get_atoms() if x.get_id() == "CA"]
    sup = Superimposer(); sup.set_atoms(ca1, ca2)
    return sup.rms

def clean_a3m(path):
    out_lines = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                out_lines.append(line.strip())
            else:
                cleaned = ''.join([c for c in line.strip() if not c.islower()])
                out_lines.append(cleaned)
    cleaned_path = str(path).replace(".a3m", "_clean.fasta")
    with open(cleaned_path, "w") as out:
        out.write("\n".join(out_lines) + "\n")
    return cleaned_path

# ─── Pipeline ─────────────────────────────────────────────────────────────────
for fasta in FASTA_DIR.glob("*.fasta"):
    name = fasta.stem
    base_dir  = RESULTS_ROOT / name
    msas_dir  = base_dir / "msas"
    pert_dir  = base_dir / "perturbed"
    msas_dir.mkdir(parents=True, exist_ok=True)
    pert_dir.mkdir(exist_ok=True)

    logger.info(f"[{name}] Downloading weights…")
    download_alphafold_params(model_type="AlphaFold2-ptm")

    logger.info(f"[{name}] Running baseline…")
    out_base = base_dir / "baseline"; out_base.mkdir(exist_ok=True)
    if fasta.stat().st_size == 0:
        logger.warning(f"[{name}] Skipping: FASTA file is empty.")
        continue
    try:
        queries, is_cplx = get_queries(str(fasta))
    except Exception as e:
        logger.error(f"[{name}] FASTA parsing failed: {e}")
        continue
    run(
        queries=queries,
        result_dir=out_base,
        use_templates=USE_TEMPLATES,
        custom_template_path=None,
        num_models=NUM_MODELS,
        is_complex=is_cplx,
        save_msa=True,
        save_recycles=True,
        save_all=True,
        save_pdb=True
    )

    msa_file = next(out_base.rglob("*.a3m"), None) or next(out_base.glob("*.sto"), None)
    if not msa_file:
        logger.error(f"[{name}] No MSA under {out_base}")
        continue
    msa_copy = msas_dir / msa_file.name
    msa_copy.write_bytes(msa_file.read_bytes())
    depth = sum(1 for line in open(msa_copy) if line.startswith(">"))
    logger.info(f"[{name}] Copied MSA → {msa_copy} (depth={depth})")

    pdb_base = next(out_base.rglob("*rank_*.pdb"), None)
    if not pdb_base:
        logger.error(f"[{name}] No *rank_*.pdb under {out_base}")
        continue

    AMINO = list("ACDEFGHIKLMNPQRSTVWY")

    def perturb_delete(src, dst):
        aln = AlignIO.read(clean_a3m(src), "fasta")
        seqs = list(aln)
        drop = random.sample(range(len(seqs)), min(NUM_DELETE, len(seqs) - 1))
        new = [s for i, s in enumerate(seqs) if i not in drop]
        if not new:
            logger.warning(f"[{name}] All sequences deleted for {dst.name}, skipping.")
            return
        temp_path = dst.with_suffix(".fasta")
        AlignIO.write(MultipleSeqAlignment(new), str(temp_path), "fasta")
        temp_path.rename(dst)
        logger.info(f"[{name}] delete → {dst}")

    def perturb_mutate(src, dst):
        aln = AlignIO.read(clean_a3m(src), "fasta")
        out = []
        for r in aln:
            seq = list(str(r.seq))
            for i in range(len(seq)):
                if random.random() < MUT_RATE:
                    seq[i] = random.choice([a for a in AMINO if a != seq[i]])
            out.append(SeqRecord(r.seq.__class__("".join(seq)), id=r.id, description=""))
        if not out:
            logger.warning(f"[{name}] No sequences to mutate for {dst.name}, skipping.")
            return
        temp_path = dst.with_suffix(".fasta")
        AlignIO.write(MultipleSeqAlignment(out), str(temp_path), "fasta")
        temp_path.rename(dst)
        logger.info(f"[{name}] mutate → {dst}")

    def perturb_shuffle(src, dst):
        aln = AlignIO.read(clean_a3m(src), "fasta")
        rows = list(aln)
        random.shuffle(rows)
        mat = np.array([list(str(r.seq)) for r in rows])
        idx = np.arange(mat.shape[1])
        np.random.shuffle(idx)
        shuffled = mat[:, idx]
        out = []
        for i, r in enumerate(rows):
            out.append(SeqRecord(type(r.seq)("".join(shuffled[i])), id=r.id, description=""))
        temp_path = dst.with_suffix(".fasta")
        AlignIO.write(MultipleSeqAlignment(out), str(temp_path), "fasta")
        temp_path.rename(dst)
        logger.info(f"[{name}] shuffle → {dst}")

    strat_list = [
        ("delete", pert_dir / f"{name}_del{NUM_DELETE}.a3m"),
        ("mutate", pert_dir / f"{name}_mut{int(MUT_RATE*100)}.a3m"),
        ("shuffle", pert_dir / f"{name}_shuffle.a3m")
    ]

    perturb_delete(msa_copy, strat_list[0][1])
    perturb_mutate(msa_copy, strat_list[1][1])
    perturb_shuffle(msa_copy, strat_list[2][1])

    # 1) Summary CSV
    csv_path = base_dir / "summary.csv"
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["strategy", "param", "depth", "mean_pLDDT", "frac>70", "frac>80", "frac>90", "RMSD"])
        pl0, f70_0, f80_0, f90_0 = extract_plddt(pdb_base)
        w.writerow(["baseline", "-", depth, f"{pl0.mean():.2f}", f"{f70_0:.2f}", f"{f80_0:.2f}", f"{f90_0:.2f}", "0.00"])
        for strat, msa in strat_list:
            outd = base_dir / strat; outd.mkdir(exist_ok=True)
            logger.info(f"[{name}] rerunning {strat}…")
            strat_fasta = base_dir / f"{name}_{strat}.fasta"
            seq = open(fasta).read().splitlines()[1]
            with open(strat_fasta, "w") as sf:
                sf.write(f">{name}_{strat}\n{seq}\n")
            q2, c2 = get_queries(str(strat_fasta))
            run(
                queries=q2,
                result_dir=outd,
                use_templates=USE_TEMPLATES,
                custom_msa_path=str(msa),
                custom_template_path=None,
                num_models=NUM_MODELS,
                is_complex=c2,
                save_msa=False,
                save_all=True,
                zip_results=False
            )
            pdb_p = next(outd.rglob("*rank_*.pdb"), None)
            if not pdb_p:
                logger.error(f"[{name}] No *rank_*.pdb under {outd}")
                continue
            plp, f70, f80, f90 = extract_plddt(pdb_p)
            rms = compute_rmsd(pdb_base, pdb_p)
            w.writerow([strat, msa.name, depth, f"{plp.mean():.2f}", f"{f70:.2f}", f"{f80:.2f}", f"{f90:.2f}", f"{rms:.3f}"])
    logger.info(f"[{name}] summary.csv saved")

    # 2) 2D Plot
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    fig, ax1 = plt.subplots()
    ax1.bar(data["strategy"], data["mean_pLDDT"].astype(float), alpha=0.6)
    ax1.set_ylabel("mean pLDDT")
    ax2 = ax1.twinx()
    ax2.plot(data["strategy"], data["RMSD"].astype(float), "ro-")
    ax2.set_ylabel("RMSD (Å)")
    plt.title(f"{name}: pLDDT vs RMSD")
    fig.savefig(base_dir / "summary_plot.png")
    plt.close()
    logger.info(f"[{name}] 2D plot saved")

    # 3) Scatterplot: MSA depth vs RMSD
    try:
        fig, ax = plt.subplots()
        depths     = data["depth"].astype(float)
        rmsd_vals  = data["RMSD"].astype(float)
        for i, strat in enumerate(data["strategy"]):
            ax.scatter(depths[i], rmsd_vals[i], label=strat)
        ax.set_xlabel("MSA depth (# sequences)")
        ax.set_ylabel("RMSD (Å)")
        ax.set_xscale("log")
        ax.set_title(f"{name}: Depth vs RMSD")
        ax.legend()
        fig.savefig(base_dir / "depth_vs_rmsd_scatter.png")
        plt.close()
        logger.info(f"[{name}] depth_vs_rmsd_scatter.png saved")
    except Exception as e:
        logger.warning(f"[{name}] Depth scatterplot failed: {e}")


    logger.info(f"[{name}] Done → {base_dir}/")
