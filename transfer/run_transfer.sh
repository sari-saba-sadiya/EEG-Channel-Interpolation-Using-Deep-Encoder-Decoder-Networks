#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --exclusive # Reserving for exclusive use
#SBATCH --mem=350G
#SBATCH --partition=sched_mit_rgmark
#SBATCH --time=96:00:00
#SBATCH --output=out_trans/trans_%a.out
#SBATCH --error=err_trans/trans_%a.err
#SBATCH --array=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sadiyasa@msu.edu

# THIS IS IMPORTANT OR THE MODULES WILL NOT IMPORT
. /etc/profile.d/modules.sh
module load python/3.6.3
module load cuda/8.0
module load cudnn/6.0

pip3 install --user virtualenv
#conda install --user virtualenv
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
KERAS_BACKEND=tensorflow
python3 ecr_transfer.py
