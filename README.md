![MSAffect Logo](logo.png)
# MSAffect

Aarav H. Dave[^1]

## Summary
This repository implements MSAffect, a computational pipeline for generating, perturbing, and analyzing multiple-sequence alignments (MSAs) in the context of protein structure prediction with AlphaFold2. It automates:  
- Baseline MSA generation via ColabFold/MMseqs2  
- Three adversarial perturbations (deletion, residue-level mutation, row-and-column shuffle)  
- AlphaFold2 runs on both unperturbed and perturbed MSAs  
- Extraction of confidence scores (pLDDT) and structural deviation metrics (RMSD)  
- Generation of summary tables, bar plots, scatterplots, and 3D visualizations

This repository hosts project files (2025.1) for public usage dictated by its license.

## Abstract
The increasing reliance on multiple sequence alignments (MSAs) for protein structure prediction necessitates a deeper understanding of how MSA perturbations impact model robustness and downstream structural inference. Current prediction pipelines often lack systematic benchmarking tools to assess sensitivity to alignment quality. This project introduces MSAffect, an end-to-end computational framework for generating, perturbing, and evaluating MSAs to probe the influence of alignment changes on AlphaFold-style structure prediction. MSAffect automates baseline MSA construction and implements three adversarial perturbation strategies: homolog deletion, residue-level substitution, and combined row/column shuffling. Resulting structural predictions are analyzed via per-residue confidence scores (pLDDT), global RMSD, and novel column-occlusion attribution maps that pinpoint alignment positions most influential to model confidence. Further enhancing MSAffect's capabilities are modules for adversarial MSA search using a genetic algorithm, latent space sampling with a variational autoencoder, and a reinforcement learning environment for learned MSA editing. Case studies on canonical soluble proteins demonstrate that even minor MSA perturbations can induce significant drops in predicted confidence and measurable conformational shifts, revealing critical loci of model brittleness. MSAffect, with its accompanying source code, examples, and documentation, provides a reproducible and extensible toolkit for robustness benchmarking and mechanistic interrogation of sequence-context effects, paving the way for more reliable and interpretable protein structure prediction.

## Paper
*Coming Soon*

## Installation
1. Open `MSAffect.ipynb` in a Google Colaboratory environment.
2. Install dependencies by running “Initiate MSAffect Environment” cell, which will install ColabFold, Biopython, NumPy, Matplotlib, and py3Dmol.
3. Prepare your FASTA inputs by placing one or more `.fasta` files in the `msa/` directory. `ubiquitin.fasta`, `bpti.fasta`, and `calmodulin.fasta` are included as examples.
4. Execute the pipeline by running the “Run MSAffect” cell. The script will process every FASTA in `msa/`, create subfolders under `results/`, and produce summary tables and figures. You can visualize your results in 3D by running the "3D Visualize Results" cell.
5. Contribute edits by adding your GitHub token and running the "Push to MSAffect GitHub" cell.

## Recognitions
*Coming Soon*

## Technical Specifications
This software utilizes:  
- Google Colaboratory
- ColabFold (AlphaFold2)
- Python 3.11
- Biopython
- NumPy
- Matplotlib
- py3Dmol
- Requests
- PyTorch
- Gym
- Stable Baselines3
- Scikit-learn

## Contacts
For questions concerning the contents of this repository, please contact contact [at] aaravdave [dot] org.

[^1]: Lowndes High School, Valdosta, GA








