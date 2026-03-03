#!/bin/bash -e
##SBATCH --partition=aoraki,aoraki_bigmem,aoraki_long,aoraki_small
#SBATCH --partition=aoraki
#SBATCH --job-name=XCP_D_OREGON_SINGLE_CENSOR_MT_redo_FS2
##SBATCH --nodelist=aoraki14,aoraki15,aoraki17
#SBATCH --exclude=aoraki23
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=2
#SBATCH --account=scoki64p
#SBATCH --profile task
#SBATCH --output=/projects/sciences/psychology/narunpat-lab/fMRIPREP/OREGON/logs/%x/%x_%j_%a.out
#SBATCH --mail-user=jack.scott@otago.ac.nz
#SBATCH --mail-type=ARRAY_TASKS
##SBATCH --array=0-217
#SBATCH --array=4,13,64,67,68,92,94,114,120,122,156,170,179,200,201


# sleep randomly for between 1 second and two minutes to stop lots of jobs starting at the same time
sleep $((RANDOM%59+1))

subjects_string=`cat /projects/sciences/psychology/narunpat-lab/fMRIPREP/OREGON/subjects/subjects_one_single.csv`

IFS=',' read -r -a array <<< "$subjects_string"

thousand="1000"

place=`expr ${SLURM_ARRAY_TASK_ID} + $thousand`

participant_id="${array[$place]}"

echo "participant id $participant_id"

work_dir_abs_path="/projects/sciences/psychology/narunpat-lab/fMRIPREP/work_dir/XCP_D_$participant_id"

export APPTAINER_BINDPATH="/projects/sciences/psychology/narunpat-lab/fMRIPREP/:/sing/"

image_path="/projects/sciences/psychology/narunpat-lab/fMRIPREP/images/xcp_d-0.10.6.simg"

in_path="/sing/data/OREGON_ONE_SINGLE/derivatives/fmriprep_24_0_0"

out_path="/sing/data/OREGON_ONE_SINGLE/derivatives/xcp_d-0.10.6_acompcor_censorMT_redoMay_12"

license_path="/sing/fs_licence/license.txt"

work_dir_path="/sing/work_dir/XCP_D_$participant_id"

rm -rf ${work_dir_abs_path}

mkdir ${work_dir_abs_path}

trap "rm -rf $work_dir_abs_path" EXIT

singularity run --cleanenv ${image_path} \
${in_path}  \
${out_path} \
participant --participant_label ${participant_id} \
--clean-workdir \
--mode linc \
--input-type fmriprep \
--file-format cifti \
--smoothing 0 \
--despike y \
--lower-bpf 0.00 \
--upper-bpf 0.08 \
--nuisance-regressors acompcor \
--dummy-scans auto \
--fd-thresh 0.5 \
--min-time 0 \
--min-coverage 0.4 \
--linc-qc y \
--abcc-qc y \
--motion-filter-type none \
--output-type auto \
--fs-license-file ${license_path} \
--combine-runs y \
--warp-surfaces-native2std y \
--notrack \
--nthreads 2 \
--mem_gb 78 \
-w ${work_dir_path}

echo "finished $participant_id , with xcp_d_compcor"

echo "finished $participant_id , removing working directory"
rm -rf ${work_dir_abs_path}
