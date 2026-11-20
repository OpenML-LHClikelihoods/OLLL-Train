#!/bin/bash
#SBATCH --job-name=olll-sleptons-lowdelta-sweep
#SBATCH --output=/ceph/hpc/home/maselekr/slurm_logs/olll-sleptons-lowdelta-sweep-%A-%a.out
#SBATCH --error=/ceph/hpc/home/maselekr/slurm_logs/olll-sleptons-lowdelta-sweep-%A-%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --mem-per-cpu=2000M
#SBATCH --cpus-per-task=24
#SBATCH --array=0-79

# Targeted Sleptons sweep near delta = nLL(mu=1) - nLL(mu=0) = 0 (no signal
# uncertainty, cards/1911.12606-nounc-sleptons.yaml) - same method as the
# EWKinos version (ewkinos_lowdelta_sweep.sh), but Sleptons' own data is
# shaped very differently: its typical-set mode sits at delta~6-7 (not
# EWKinos' ~22-25), and [2,4) is already reasonably populated (~8.5% of the
# dataset) - the genuinely thin band is [0,2) ([0,0.25): 0.12%, [0.25,1):
# 0.86%, [1,2): 2.71%, 2026-11-20 analysis), so that is the target here,
# not [0,4).
#
# start_method='fine-tune' (utils.generate_starting_points, set in
# parameters_sleptons_lowdelta.yaml) draws each chain's start from a narrow
# Gaussian centred at zero signal. points=40 was picked from a local
# calibration smoke test (2026-11-20, 4 chains, same SR_sigma=CR_sigma=0.01
# as the original production Sleptons no-unc run used): mean delta_exp grows
# from ~0.05 at step 0 to ~1.3-1.5 by step 35-39, landing inside the target
# [0,2) band without drifting past it.
#
# A points=40 chain is not "short" (same order of magnitude as the EWKinos
# points=50 chain, measured ~48 min/chain locally there), so scans=24
# matches --cpus-per-task=24 - one chain per worker, not several run back-
# to-back. The cluster's own per-chain cost for this hasn't been measured
# yet (the local smoke test ran under contention from a separate 24-core
# validation job), so --time=02:00:00 is a provisional budget, same
# reasoning as the EWKinos sweep's own first --time guess - watch the first
# completed tasks' .err logs for CANCELLED ... DUE TO TIME LIMIT before
# trusting it for the full array.
#
# With `cluster : True` set in parameters_sleptons_lowdelta.yaml, sample.py
# still adds the task ID to the random seed and writes to
# <output_folder>/<analysis>-<n>-<task ID>, so tasks never collide even
# though every task uses the same fine-tune start method.
#
# --array=0-79 x scans=24 x points=40 -> ~76,800 rows total, concentrated
# in [0, ~1.5) (same order of magnitude as the EWKinos fix's ~74,700 rows).

set -eo pipefail   # no -u: conda's activate scripts reference unset variables

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/ceph/hpc/home/maselekr/projects/OLLL-Train"
SAMPLING_DIR="${PROJECT_DIR}/sampling"
PARAM_FILE="${SAMPLING_DIR}/parameters_sleptons_lowdelta.yaml"
DATA_DIR="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/tables"   # must match output_folder in parameters_sleptons_lowdelta.yaml
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
# parameters_sleptons_lowdelta.yaml uses paths relative to sampling/ (../data/, ../tables/)
cd "${SAMPLING_DIR}"
START_MARKER=$(mktemp)
python sample.py "${PARAM_FILE}" --log_dir "${LOG_DIR}"

# ── Verify output ─────────────────────────────────────────────────────────────
# Only results-*.csv proves sample.py reached its final merge step; a killed
# or still-running task leaves table-*.csv (+ metadata.json) behind too, so
# matching any *.csv here would report success for an incomplete run.
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
