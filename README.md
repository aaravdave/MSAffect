# MSAffect

Aarav H. Dave[^1]

## Summary
This repository implements **MSAffect**, a computational pipeline for generating, perturbing, and analyzing multiple-sequence alignments (MSAs) in the context of protein structure prediction with AlphaFold2. It automates:  
- Baseline MSA generation via ColabFold/MMseqs2  
- Three adversarial perturbations (deletion, residue-level mutation, row-and-column shuffle)  
- AlphaFold2 runs on both unperturbed and perturbed MSAs  
- Extraction of confidence scores (pLDDT) and structural deviation metrics (RMSD)  
- Generation of summary tables, bar plots, scatterplots, and 3D visualizations

This repository hosts project files (2025.1) for public usage dictated by its license.

## Abstract
MSAffect is an end-to-end computational pipeline that generates, perturbs, and evaluates multiple-sequence alignments (MSAs) to probe the sensitivity of protein structure prediction to alignment changes. The package automates baseline MSA construction and three adversarial perturbation strategies—homolog deletion, residue-level substitution, and combined row/column shuffling—followed by AlphaFold-style structure prediction. Outputs are analysed with per-residue confidence scores (pLDDT), global structural deviation (RMSD), and column-occlusion attribution maps that highlight alignment positions most influential to model confidence. Complementary modules implement a genetic algorithm for adversarial MSA search, a compact variational autoencoder for latent-space sampling, and a Gym-compatible reinforcement-learning environment for learned editing policies. Optional dependencies are detected at runtime; a dry-run simulation mode fabricates lightweight MSA and PDB artifacts to accelerate development and reproducible testing. Case studies on canonical soluble proteins demonstrate that modest MSA perturbations can precipitate marked drops in predicted confidence and measurable conformational shifts, revealing loci of model brittleness. MSAffect provides a reproducible, extensible toolkit for systematic robustness benchmarking and mechanistic interrogation of sequence-context effects on structural prediction. Source code, examples, and documentation accompany this public release.

## Paper
*Coming Soon*

## Installation
1. **Open `MSAffect.ipynb`** in a Google Colaboratory environment.
2. **Install dependencies** by running “Initiate MSAffect Environment” cell, which will install ColabFold, Biopython, NumPy, Matplotlib, and py3Dmol.
3. **Prepare your FASTA inputs** by placing one or more `.fasta` files in the `msa/` directory. `ubiquitin.fasta`, `bpti.fasta`, and `calmodulin.fasta` are included as examples.
4. **Execute the pipeline** by running the “Run MSAffect” cell. The script will process every FASTA in `msa/`, create subfolders under `results/`, and produce summary tables and figures. You can visualize your results in 3D by running the "3D Visualize Results" cell.
5. **Contribute edits** by adding your GitHub token and running the "Push to MSAffect GitHub" cell.

## Recognitions
*Coming Soon*

## Technical Specifications
This software utilizes:  
- **Google Colaboratory** for interactive execution and free GPU access  
- **ColabFold** (AlphaFold2) for accelerated MSA generation and structure prediction  
- **Python 3.11** as the execution environment  
- **Biopython** for MSA parsing and PDB handling  
- **NumPy** for numerical operations  
- **Matplotlib** (and optional Seaborn) for plotting  
- **py3Dmol** for in-notebook 3D molecular visualization  

## License
This software, as with all subsequent versions of the software, is protected by the CC-BY-NC-ND license. In summary, this does not allow commercial usage, distribution, or distribution of modifications of the software. In additon, you are required to credit authorship and state any changes you may have made.
> For more information, please refer to the `LICENSE` file.

## Contacts
For questions concerning the contents of this repository, please contact contact [at] aaravdave [dot] org.

[^1]: Lowndes High School, Valdosta, GA


