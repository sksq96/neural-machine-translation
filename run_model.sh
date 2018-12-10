#!/bin/sh

#SBATCH --job-name=nmt
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB
#SBATCH --time=30:0:00
#SBATCH --output=./slurm-output/nmt-slurm.%j.out
#SBATCH --gres=gpu:1
##SBATCH --nodelist=lion21
##SBATCH --partition=v100_sxm2_4, v100_pci_2
##SBATCH --exclusive
##SBATCH --res=morari 

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi
alias p3="python3"
alias p="pyenv"
p shell anaconda3-5.0.0


ROOT="/home/sc7268/neural-machine-translation"
cd $ROOT

RUN="$1"
SAVE=$ROOT/logs/$RUN
mkdir -p $SAVE

eval "python -u $ROOT/train.py --batch_size 16 | tee $SAVE/exp_";

