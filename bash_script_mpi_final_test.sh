#!/bin/bash
#SBATCH --job-name=my_mpi_final_test_job
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=rome
#SBATCH --nodes=2
#SBATCH --cpus-per-task=128
#SBATCH --ntasks=2
#SBATCH --mem=100G
#SBATCH --time=00:05:00
#SBATCH --ntasks-per-node=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export UCX_WARN_UNUSED_ENV_VARS=n 


module purge
module load gcc
module load openmpi

make clean && make mpi_omp

export LD_LIBRARY_PATH=$HOME/local/matio/lib:$HOME/local/hdf5/lib:$HOME/local/zlib/lib:$LD_LIBRARY_PATH

INPUT_FILE_NAME="mawi_201512020330.mat"
SOURCE_FILE="$HOME/ergasia2_parallhla/ergasia1_source_code/matrices/$INPUT_FILE_NAME"
JOB_WORKING_DIR="/scratch/d/dimopoul/$SLURM_JOB_ID"

mkdir -p "$JOB_WORKING_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

echo "Copying $SOURCE_FILE to $JOB_WORKING_DIR/$INPUT_FILE_NAME..."
/usr/bin/cp "$SOURCE_FILE" "$JOB_WORKING_DIR/$INPUT_FILE_NAME"

if [ $? -ne 0 ]; then
    echo "ERROR: File copy failed. Check source path or file size."
    rm -rf "$JOB_WORKING_DIR"
    exit 1
fi

/usr/bin/cp ./mpi_omp "$JOB_WORKING_DIR/mpi_omp"

echo "Staging completed. Starting MPI OpenMP program..."

cd "$JOB_WORKING_DIR"

mpiexec -n $SLURM_NTASKS ./mpi_omp "$JOB_WORKING_DIR/$INPUT_FILE_NAME"
rm -rf "$JOB_WORKING_DIR"
