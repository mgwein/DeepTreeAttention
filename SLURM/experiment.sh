#!/bin/bash
#SBATCH --job-name=DeepTreeAttention   # Job name
#SBATCH --mail-type=ALL               # Mail events
#SBATCH --mail-user=mgwein@ufl.edu  # Where to send mail
#SBATCH --account=azare
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/blue/azare/mgwein/DeepTreeAttention/DeepTreeAttention_logs/DeepTreeAttention_%j.out   # Standard output and error log
#SBATCH --error=/blue/azare/mgwein/DeepTreeAttention/DeepTreeAttention_logs/DeepTreeAttention_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1

ulimit -c 0

module load git gcc

git checkout $1

source activate DeepTreeAttention

# cd ~/DeepTreeAttention/

#get branch and commit name
branch_name=$((git symbolic-ref HEAD 2>/dev/null || echo "(unnamed branch)")|cut -d/ -f3-)
commit=$(git log --pretty=format:'%H' -n 1)
python train.py $branch_name $commit
