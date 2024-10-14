#!/bin/bash
#SBATCH --account jsp_student_projects
#SBATCH -c 8
#SBATCH --mem 64g
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --time 7-00:00:00

python3 mirna_sweep.py