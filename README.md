This project implements a hybrid MPI + OpenMP algorithm for computing connected components on large sparce matrices
THe input matrices are provided fomr Suitsparse collection repository, in MATLAB .mat format and are read using the matio library
The code was developed and evaluated on the Aristotle HPC cluster using Slurm

DEPENDECIES:
  Required software:
    GCC (with OpenMP support)
    MPI implementation (OpenMPI)
    matio library
    matio dependecies (zlib, HDF5)

DEPENDECY INSTALLATION NOTE (IMPORTANT):
  On the Aristotle cluster, the matio lirbary and its dependecies were not available system-wide:
  ThereFore, the following libraries were installed locally by me:
    $HOME/local/zlib
    $HOME/local/hdf5
    $HOME/local/matio
The code will not run unless these libraries are available and properly linked

Build the MPI+OpenMP executable:
  make mpi_omp

This will compile 'main_mpi_openmp.c' and 'coloring_mpi_openmp.c' into the executable 'mpi_omp'

Note:
  The Makefile assumes the matio libray and its dependecies (HDF5, zlib) are installed locally in $HOME/local
  In the bash script i set the path needed.

Next up, in the command line type: sbatch bash_script_mpi_final_test.sh, or whatever other slurm command for running a job, and everything happens automatically.
The .out and .err files are created. On the .out file is printed the output and on the .err file, the errors, if any happened to appear.
On the bash script, schange the INPUT_FILE_NIME according to the file that you want the code to run on. 
Also, the example matrix, must be installed on a "matrices" directory, because of the paths that i've set on bash script.
You can modify that scirpt the way you like.
