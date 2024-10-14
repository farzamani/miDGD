#!/bin/bash
#SBATCH --account jsp_student_projects
#SBATCH -c 16
#SBATCH --mem 128g
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --time 3-00:00:00

python3 midgd_cv.py