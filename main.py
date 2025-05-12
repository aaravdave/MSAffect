#!/usr/bin/env python3
"""
MSAffect: Adversarial Sequence Alignments for Protein Structure Prediction
Fully implements environment checks, MSA generation (with fallback), perturbations, runs, analyses.
"""
import os
import sys
import random
import subprocess
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, Superimposer

# Seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Environment verification
def check_environment():
    if sys.version_info < (3,8):
        sys.exit("Python 3.8+ is required.")
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if props.total_memory < 16 * 1024**3:
                logging.warning("GPU memory <16GB detected.")
        else:
            logging.warning("CUDA GPU not detected; performance may be degraded.")
    except ImportError:
        logging.warning("PyTorch not installed; cannot check GPU.")

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(asctime)s] %(message)s")

# Paths and parameters
MSA_DIR = Path("msa")
RESULTS_ROOT = Path("results")
NUM_MODELS = 5
USE_TEMPLATES = False
MUT_RATE = 0.02
NUM_DELETE = 5

RESULTS_ROOT.mkdir(exist_ok=True)

# Helper: clean A3M to uniform FASTA
def clean_a3m(a3m_path):
    out = a3m_path.with_suffix('.clean.fasta')
    lines = []
    with open(a3m_path) as f:
        for line in f:
            if line.startswith('>'):
                seq_id = line.strip()
                lines.append(seq_id)
            else:
                seq = ''.join([c for c in line.strip() if c.isupper() or c=='-'])
                lines.append(seq)
    with open(out, 'w') as w:
        w.write('\n'.join(lines) + '\n')
    return out

# MSA via MMseqs2 fallback
def run_mmseqs2(fasta, output_prefix):
    tmp_db = output_prefix + "_db"
    uniref = "uniref90"
    try:
        subprocess.run(["mmseqs","createdb",fasta,tmp_db],check=True)
        subprocess.run(["mmseqs","search",tmp_db,uniref,tmp_db+"_res","/tmp"],check=True)
        subprocess.run(["mmseqs","convertalis",tmp_db,uniref,tmp_db+"_res",output_prefix+".m8"],check=True)
        subprocess.run(["mmseqs","result2msa",tmp_db,uniref,tmp_db+"_res",output_prefix+".a3m"],check=True)
        return output_prefix+".a3m"
    except Exception:
        logging.error("MMseqs2 unavailable or failed; fallback")
        return None

# Structure extraction
def extract_plddt(pdb):
    struct = PDBParser(QUIET=True).get_structure('m',str(pdb))
    b = [a.get_bfactor() for a in struct.get_atoms() if a.get_id()=='CA']
    arr = np.array(b)
    return arr, (arr>70).mean(), (arr>80).mean(), (arr>90).mean()

def compute_rmsd(p1,p2):
    p = PDBParser(QUIET=True)
    s1 = p.get_structure('a',str(p1))
    s2 = p.get_structure('b',str(p2))
    ca1=[a for a in s1.get_atoms() if a.get_id()=='CA']
    ca2=[a for a in s2.get_atoms() if a.get_id()=='CA']
    sup=Superimposer(); sup.set_atoms(ca1,ca2)
    return sup.rms

def tm_score(p1,p2):
    try:
        r=subprocess.run(["TMscore",str(p1),str(p2)],capture_output=True,text=True)
        for l in r.stdout.splitlines():
            if l.startswith('TM-score='): return float(l.split()[1])
    except:
        logging.warning('TMscore failed')
    return np.nan

# Perturbation routines using cleaned alignment

def perturb_delete(msa,a3m_out,n):
    clean=clean_a3m(Path(msa))
    aln=AlignIO.read(str(clean),'fasta')
    if len(aln)<=n:
        logging.warning('skip delete')
        return False
    keep=[r for i,r in enumerate(aln) if i not in random.sample(range(len(aln)),n)]
    AlignIO.write(MultipleSeqAlignment(keep),str(a3m_out),'fasta')
    return True


def perturb_mutate(msa,a3m_out,rate):
    clean=clean_a3m(Path(msa))
    aln=AlignIO.read(str(clean),'fasta')
    AMINO=list('ACDEFGHIKLMNPQRSTVWY')
    out=[]
    for r in aln:
        seq=list(str(r.seq))
        for i in range(len(seq)):
            if random.random()<rate: seq[i]=random.choice([a for a in AMINO if a!=seq[i]])
        out.append(SeqRecord(type(r.seq)(''.join(seq)),id=r.id,description=''))
    AlignIO.write(MultipleSeqAlignment(out),str(a3m_out),'fasta')
    return True


def perturb_shuffle(msa,a3m_out,_):
    clean=clean_a3m(Path(msa))
    aln=AlignIO.read(str(clean),'fasta')
    rows=list(aln); random.shuffle(rows)
    mat=np.array([list(str(r.seq)) for r in rows])
    idx=np.arange(mat.shape[1]); np.random.shuffle(idx)
    sh=mat[:,idx]; out=[]
    for i,r in enumerate(rows):
        out.append(SeqRecord(type(r.seq)(''.join(sh[i])),id=r.id,description=''))
    AlignIO.write(MultipleSeqAlignment(out),str(a3m_out),'fasta')
    return True

# Run colabfold

def run_cf(fasta,msa,odir):
    from colabfold.batch import run,get_queries
    q,c=get_queries(fasta)
    run(queries=q,result_dir=odir,use_templates=USE_TEMPLATES,custom_msa_path=msa,num_models=NUM_MODELS,is_complex=c,save_pdb=True,save_msa=False)

# Pipeline
if __name__=='__main__':
    check_environment()
    for f in MSA_DIR.glob('*.fasta'):
        n=f.stem; b=RESULTS_ROOT/n; m=b/'msas'; p=b/'perturbed'
        for d in (b,m,p): d.mkdir(parents=True,exist_ok=True)
        # baseline MSA
        pre=m/(n+'_baseline'); a3m=run_mmseqs2(str(f),str(pre))
        if not a3m:
            ob=b/'baseline'; ob.mkdir(exist_ok=True)
            from colabfold.download import download_alphafold_params
            download_alphafold_params('AlphaFold2-ptm')
            from colabfold.batch import run,get_queries
            q,c=get_queries(str(f))
            run(queries=q,result_dir=ob,use_templates=USE_TEMPLATES,custom_msa_path=None,num_models=NUM_MODELS,is_complex=c,save_pdb=False,save_msa=True)
            a3m=next(ob.rglob('*.a3m'))
        depth=sum(1 for l in open(a3m) if l.startswith('>'))
        # baseline structure
        ob=b/'baseline'; ob.mkdir(exist_ok=True)
        from colabfold.download import download_alphafold_params
        download_alphafold_params('AlphaFold2-ptm')
        from colabfold.batch import run,get_queries
        q,c=get_queries(str(f))
        run(queries=q,result_dir=ob,use_templates=USE_TEMPLATES,custom_msa_path=a3m,num_models=NUM_MODELS,is_complex=c,save_pdb=True,save_msa=False)
        p0=next(ob.rglob('*.pdb')); arr0,f70_0,f80_0,f90_0=extract_plddt(p0)
        # perturb
        strat=[('delete',pre.with_name(n+'_delete.a3m'),NUM_DELETE),('mutate',pre.with_name(n+'_mut.a3m'),MUT_RATE),('shuffle',pre.with_name(n+'_shuffle.a3m'),None)]
        recs=[{'strategy':'baseline','param':'-','depth':depth,'mean_pLDDT':arr0.mean(),'frac>70':f70_0,'frac>80':f80_0,'frac>90':f90_0,'RMSD':0,'TM-score':1}]
        for s,path,par in strat:
            ok=globals()[f'perturb_{s}'](a3m,path,par)
            if not ok: continue
            od=b/s; od.mkdir(exist_ok=True)
            run_cf(str(f),str(path),str(od))
            pp=next(Path(od).rglob('*.pdb')); arr,f70,f80,f90=extract_plddt(pp); rms=compute_rmsd(p0,pp); tm=tm_score(p0,pp)
            recs.append({'strategy':s,'param':par,'depth':depth,'mean_pLDDT':arr.mean(),'frac>70':f70,'frac>80':f80,'frac>90':f90,'RMSD':rms,'TM-score':tm})
        df=pd.DataFrame(recs); df.to_csv(b/'summary.csv',index=False); df.to_pickle(b/'summary.pkl')
        # plots
        plt.figure(); df.plot.bar(x='strategy',y='mean_pLDDT',yerr=df['mean_pLDDT'].std(),legend=False); plt.ylabel('Mean pLDDT'); plt.title(f'{n}'); plt.savefig(b/'pLDDT_bar.png'); plt.close()
        plt.figure(); plt.scatter(df['mean_pLDDT'],df['RMSD']); m,b_=np.polyfit(df['mean_pLDDT'],df['RMSD'],1); plt.plot(df['mean_pLDDT'],m*df['mean_pLDDT']+b_); plt.xlabel('Mean pLDDT'); plt.ylabel('RMSD'); plt.savefig(b/'RMSD_vs_pLDDT.png'); plt.close()
        logging.info(f'{n} complete')
