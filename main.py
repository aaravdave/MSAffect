import os
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import PDBParser, Superimposer

# Local ColabFold imports
from colabfold.batch import run, get_queries
from colabfold.download import download_alphafold_params

# ======================== Logging Setup ========================
import logging
from datetime import datetime
from termcolor import colored

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Format the timestamp
        time = datetime.fromtimestamp(record.created).strftime("[%H:%M:%S]")
        time_colored = colored(time, "cyan")  # or "green", "magenta", etc.

        # Format the message without INFO/WARNING
        message = record.getMessage()
        return f"{time_colored} {message}"

# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear old handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Set up console handler with our custom formatter
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())

logger.addHandler(handler)

# ======================== pLDDT Plotting ========================
def plot_plddt(plddt_scores, save_path=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 4))
    plt.plot(plddt_scores, label="pLDDT")
    plt.axhline(70, color="orange", linestyle="--", label="70 (Confident)")
    plt.axhline(90, color="green", linestyle="--", label="90 (High Confidence)")
    plt.xlabel("Residue Index")
    plt.ylabel("pLDDT Score")
    plt.title("pLDDT per Residue")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"pLDDT plot saved to {save_path}")
    plt.show()


# ======================== Local Prediction Wrapper ========================
def run_prediction_local(fasta_path, output_dir, num_models=5, use_templates=False):
    """
    Runs a local ColabFold (AlphaFold2-ptm) prediction from a FASTA, saving into output_dir.
    """
    # Download weights once
    logger.info("Ensuring AlphaFold2-ptm parameters are downloaded...")
    download_alphafold_params(model_type="AlphaFold2-ptm")

    # Prepare queries
    logger.info(f"Reading sequence from: {fasta_path}")
    queries, is_complex = get_queries(fasta_path)

    # Run prediction
    logger.info("Running ColabFold local prediction...")
    run(
        queries=queries,
        result_dir=Path(output_dir),
        use_templates=use_templates,
        custom_template_path=None,
        num_models=num_models,
        is_complex=is_complex
    )
    logger.info("Local prediction complete.")


if __name__ == "__main__":
    # === User parameters ===
    FASTA = "msa/ubiquitin.fasta"           # input FASTA
    RESULTS = "results"                     # output directory
    MODELS = 5                              # how many models to run
    USE_TEMPLATES = False                   # template usage flag

    # Clean slate
    if os.path.exists(RESULTS):
        logger.info(f"Removing existing '{RESULTS}' folder for a fresh run")
        os.system(f"rm -rf {RESULTS}")

    # Run local prediction
    run_prediction_local(FASTA, RESULTS, num_models=MODELS, use_templates=USE_TEMPLATES)

    # ================= Post-processing =================
    # Pick the top-ranked PDB (ColabFold names it rank_001_*.pdb)
    pdb_files = sorted(Path(RESULTS).glob("rank_*.pdb"))
    if not pdb_files:
        logger.error("No PDB files found in results!")
        exit(1)

    baseline_pdb = pdb_files[0]
    logger.info(f"Loading baseline PDB: {baseline_pdb}")

    # Extract per-residue pLDDT from the B-factor column
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("baseline", baseline_pdb)
    plddt_scores = [atom.get_bfactor() for atom in struct.get_atoms() if atom.get_id() == "CA"]
    plddt_scores = np.array(plddt_scores)

    # Plot and save
    plot_plddt(plddt_scores, save_path=os.path.join(RESULTS, "baseline_plddt.png"))
    logger.info(f"Baseline mean pLDDT: {plddt_scores.mean():.2f}")
