#!/bin/bash -e
#SBATCH --partition=aoraki,aoraki_bigmem,aoraki_long,aoraki_small
#SBATCH --job-name=fMRIprep_OS_OREGON
#SBATCH --time=96:00:00
#SBATCH --mem=100000MB
#SBATCH --cpus-per-task=4
#SBATCH --account=vanwi84p
#SBATCH --profile task
#sbatch --nice
#SBATCH --output=/projects/sciences/psychology/narunpat-lab/fMRIPREP/OREGON/logs/%x/%x_%j_%a.out
#SBATCH --mail-user=william.vandervliet@otago.ac.nz
#SBATCH --mail-type=ARRAY_TASKS
#SBATCH --array=578,610,631,673,675,677,684,685,691,694,700,717,734,765,788,789,808,809,835
##SBATCH --array=0-999

module load apptainer/fMRIPrep/23.2.3

echo OREGON
echo 0-1217

subjects_string=`cat /projects/sciences/psychology/narunpat-lab/fMRIPREP/OREGON/subjects/subjects_one_single.csv`

IFS=',' read -r -a array <<< "$subjects_string"

participant_id="${array[${SLURM_ARRAY_TASK_ID}]}"

work_dir_abs_path="/projects/sciences/psychology/narunpat-lab/fMRIPREP/works/${participant_id}"

export APPTAINER_BINDPATH="/projects/sciences/psychology/narunpat-lab/fMRIPREP/:/nobackup/"

image_path="/projects/sciences/psychology/narunpat-lab/fMRIPREP/images/fmriprep_24.0.0.simg"

bids_path="/nobackup/data/OREGON_ONE_SINGLE/"

out_path="/nobackup/data/OREGON_ONE_SINGLE/derivatives/fmriprep_24_0_0/"

work_dir_path="/nobackup/works/${participant_id}"

license_path="/nobackup/fs_licence/license.txt"

fs_sub_dir="/nobackup/data/OREGON_ONE_SINGLE/derivatives/freesurfer/"

rm -rf ${work_dir_abs_path}
mkdir ${work_dir_abs_path}

trap "rm -rf $work_dir_abs_path" EXIT

apptainer run --cleanenv ${image_path} \
${bids_path}  \
${out_path} \
participant \
--participant_label ${participant_id} \
--omp-nthreads 4 \
--mem_mb 100000 \
--skip_bids_validation \
--level full \
--longitudinal \
--fs-license-file ${license_path} \
-w ${work_dir_path} \
--clean-workdir \
--cifti-output \
--write-graph \
--notrack \
--fs-subjects-dir ${fs_sub_dir}

echo "finished ${participant_id}"

rm -rf ${work_dir_abs_path}
