#!/bin/bash
#SBATCH --job-name=olll-ewkinos-lowdelta-sweep
#SBATCH --output=/ceph/hpc/home/maselekr/slurm_logs/olll-ewkinos-lowdelta-sweep-%A-%a.out
#SBATCH --error=/ceph/hpc/home/maselekr/slurm_logs/olll-ewkinos-lowdelta-sweep-%A-%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --mem-per-cpu=2000M
#SBATCH --cpus-per-task=8
#SBATCH --array=0-149

# Targeted EWKinos sweep near delta = nLL(mu=1) - nLL(mu=0) = 0 (no signal
# uncertainty, cards/1911.12606-nounc-EWKinos.yaml). The usual boundary-scan
# MCMC (parameters.yaml) concentrates its samples in the high-dimensional
# "typical set" of the 44-bin random walk (delta of order tens), not near
# S=0 where delta is small - but that neighbourhood is exactly where the
# expected exclusion threshold (delta of order 1-4) sits.
#
# Unlike the earlier per-row sweeps (ewkinos_deficit_sweep_from_csv.sh,
# sleptons_sampling_from_csv.sh), this one does not read starting points
# from a CSV: parameters_ewkinos_lowdelta.yaml sets start_method='fine-tune'
# (utils.generate_starting_points), which draws each chain's start from a
# narrow Gaussian centred at zero signal.
#
# First version of this script used points=2 (one MCMC step per chain, 40
# chains/task): that only ever lands in delta in [0, 0.25) and stops, not
# the [0.25, 4) band the exclusion threshold actually sits in, since a
# single random-walk step from the S=0 start can't get further. A local
# step-by-step trace (2026-11-19) showed delta growing ~linearly with step
# count (it's a quadratic observable of the walk's displacement, so grows
# with variance, i.e. with N, not sqrt(N)) through the whole [0.25, 4) gap
# by step ~50, without yet reaching the already-dense [4, 20) bulk -
# parameters_ewkinos_lowdelta.yaml now sets points=50 so every step along
# that walk is kept as its own training row, spread across the band that
# actually needed filling, not just whichever point the chain stopped at.
#
# A points=50 chain is no longer "short": measured ~48 min/chain locally
# (vs. ~6.5 min/chain for the old points=2 version, almost all of which was
# the one-off calculate_Lmax cost paid on a chain's first step regardless
# of its length). So parameters_ewkinos_lowdelta.yaml now sets scans=8,
# matching --cpus-per-task=8 - one chain per worker, not several run back-
# to-back - and --time is bumped to 04:00:00 for margin, since the cluster's
# own per-chain cost hasn't been measured for this length yet (the earlier
# --time=00:30:00 run undershot in exactly this way: it left every task
# with table-*.csv but no results-*.csv, CANCELLED ... DUE TO TIME LIMIT).
#
# With `cluster : True` set in parameters_ewkinos_lowdelta.yaml, sample.py
# still adds the task ID to the random seed and writes to
# <output_folder>/<analysis>-<n>-<task ID>, so tasks never collide even
# though every task uses the same fine-tune start method.
#
# --array=0-149 x scans=8 x points=50 -> ~60,000 rows total, concentrated
# in [0, ~3) (old training set has only ~34,000 rows total in [0, 4)).

set -eo pipefail   # no -u: conda's activate scripts reference unset variables

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/ceph/hpc/home/maselekr/projects/OLLL-Train"
SAMPLING_DIR="${PROJECT_DIR}/sampling"
PARAM_FILE="${SAMPLING_DIR}/parameters_ewkinos_lowdelta.yaml"
DATA_DIR="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/tables"   # must match output_folder in parameters_ewkinos_lowdelta.yaml
CONDA_ENV="likelihood"               # env built from sampling/requirements.txt

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}
# sample.py names its log file by the second, so give every task its own folder
LOG_DIR="${SAMPLING_DIR}/logs/${JOB_ID}/task-${TASK_ID}"

echo "===== Job info ====="
echo "Array job ID  : ${JOB_ID}"
echo "Array task ID : ${TASK_ID}"
echo "Param file    : ${PARAM_FILE}"
echo "Output dir    : ${OUTPUT_DIR}"
echo "Log dir       : ${LOG_DIR}"
echo "Node          : $(hostname)"
echo "CPUs          : ${SLURM_CPUS_PER_TASK:-1}"
echo "Start time    : $(date)"
echo "===================="

# ── Conda ─────────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
echo "$(python --version) from $(which python)"

# Keep BLAS/TF/JAX to the CPUs Slurm gave us (TF is also pinned to 1 thread
# per process inside likelihood.py).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p /ceph/hpc/home/maselekr/slurm_logs "${OUTPUT_DIR}" "${LOG_DIR}"

# ── Unpack the workspaces once ────────────────────────────────────────────────
# sample.py unpacks <analysis>.tar.gz when data/<analysis>/ is missing; with
# many tasks starting together they would all extract into the same folder at
# once. Do it here under a lock, so only the first task extracts and the rest
# wait.
(
    flock -w 1800 9 || { echo "✗ Timed out waiting for the data unpacking lock"; exit 1; }
    for archive in "${DATA_DIR}"/*.tar.gz; do
        name=$(basename "${archive}" .tar.gz)
        if [ ! -d "${DATA_DIR}/${name}" ]; then
            echo "Unpacking ${name}.tar.gz"
            tar -xzf "${archive}" -C "${DATA_DIR}"
        fi
    done
) 9>"${DATA_DIR}/.unpack.lock"

# ── Run the sampler ────────────────────────────────────────────────────────────
# parameters_ewkinos_lowdelta.yaml uses paths relative to sampling/ (../data/, ../tables/)
cd "${SAMPLING_DIR}"
START_MARKER=$(mktemp)
python sample.py "${PARAM_FILE}" --log_dir "${LOG_DIR}"

# ── Verify output ─────────────────────────────────────────────────────────────
# Only results-*.csv proves sample.py reached its final merge step; a killed
# or still-running task leaves table-*.csv (+ metadata.json) behind too, so
# matching any *.csv here would report success for an incomplete run (this
# is exactly what happened at --time=00:30:00: tasks cut off mid-chain still
# "passed" this check because their partial table-*.csv files matched).
RESULT_CSVS=$(find "${OUTPUT_DIR}" -path "*-${TASK_ID}/results-*.csv" -newer "${START_MARKER}")
rm -f "${START_MARKER}"
if [ -n "${RESULT_CSVS}" ]; then
    echo "✓ Task ${TASK_ID} finished at $(date), wrote:"
    echo "${RESULT_CSVS}"
    exit 0
else
    echo "✗ Error: task ${TASK_ID} produced no results-*.csv in ${OUTPUT_DIR}/*-${TASK_ID}/ " \
         "(incomplete run - check for table-*.csv there and see ${LOG_DIR})"
    exit 1
fi
