GCC   = gcc
MPICC = mpicc
CFLAGS = -march=native -O2 

MATIO_BASE = $(HOME)/local

MATIO_INCLUDES = -I$(MATIO_BASE)/matio/include -I$(MATIO_BASE)/hdf5/include -I$(MATIO_BASE)/zlib/include
MATIO_LIBS     = -L$(MATIO_BASE)/matio/lib -L$(MATIO_BASE)/hdf5/lib -L$(MATIO_BASE)/zlib/lib -lmatio -lhdf5 -lz -lm

MPI_OMP = mpi_omp

all: $(MPI_OMP)

$(MPI_OMP): main_mpi_openmp.c coloringCC_mpi_openmp.c 
	$(MPICC) $(CFLAGS) $(MATIO_INCLUDES) -fopenmp $^ $(MATIO_LIBS) -o $@

clean:
	rm -f $(MPI_OMP)