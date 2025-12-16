#ifndef COLORINGCC_MPI_OPENMP_H
#define COLORINGCC_MPI_OPENMP_H
#include <mpi.h>

void coloringCC_mpi_openmp(int nrows, const int *rowptr, const int *index, int *labels, int rank, int size, MPI_Comm comm);

#endif