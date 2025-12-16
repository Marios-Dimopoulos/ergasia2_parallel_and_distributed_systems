#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <matio.h>

#include "coloringCC_mpi_openmp.h"

int main(int argc, char* argv[]) {

    struct timeval start;
    struct timeval end;

    if (argc < 2) {
        printf("Usage: %s <mat_file>\n", argv[0]);
        return 1;
    }

    mat_t *matfp = Mat_Open(argv[1], MAT_ACC_RDONLY);
    if (matfp == NULL) {
        printf("Cannot open MAT file %s\n", argv[1]);
        return 1;
    }

    matvar_t *top_level_var = NULL;
    matvar_t *matvar = NULL; 
    top_level_var = Mat_VarReadNext(matfp);
    
    if (top_level_var == NULL || top_level_var->class_type != MAT_C_STRUCT) {
        printf("ERROR: MAT file does not contain a top-level struct variable as expected.\n");
        if (top_level_var != NULL) Mat_VarFree(top_level_var);
        Mat_Close(matfp);
        return 1;
    }
    
    matvar = Mat_VarGetStructFieldByName(top_level_var, (char*)"A", 0);
    
    if (matvar == NULL) {
        printf("ERROR: Could not find the required 'A' field inside the top-level structure.\n");
        Mat_Close(matfp);
        Mat_VarFree(top_level_var);
        return 1;
    }

    if (matvar->class_type != MAT_C_SPARSE) {
        printf("ERROR: Found field 'A' is not sparse (class type %d) in MAT file.\n", matvar->class_type);
        Mat_Close(matfp);
        Mat_VarFree(top_level_var);
        return 1;
    }

    int nrows = matvar->dims[0];
    int ncols = matvar->dims[1];

    mat_sparse_t *sparse_data = (mat_sparse_t*)matvar->data;

    int *ir_original = (int*)sparse_data->ir;
    int *jc_original = (int*)sparse_data->jc;
    if (!ir_original || !jc_original) {
        printf("Error: Sparse matrix data is NULL\n");
        Mat_VarFree(top_level_var);
        Mat_Close(matfp);
        return 1;
    }

    int nnz = (int)jc_original[ncols];

    int *rowptr = calloc(nrows + 1, sizeof(int));
    int *index = malloc(nnz * sizeof(int));
    if (!rowptr || !index) {
        printf("Memory allocation failed\n");
        Mat_Close(matfp);
        free(rowptr); free(index);
        Mat_VarFree(top_level_var);
        return 1;
    }

    for (int j = 0; j < ncols; j++) {
        for (int p = jc_original[j]; p < jc_original[j+1]; p++) {
            int i = ir_original[p]; 
            if (i < nrows) {
                rowptr[i+1]++;
            }
        }
    }

    for (int i = 1; i <= nrows; i++) {
        rowptr[i] += rowptr[i-1];
    }

    int *temp_rowptr = malloc(nrows * sizeof(int));
    if (!temp_rowptr) {
        printf("Memory allocation failed\n");
        Mat_VarFree(top_level_var);
        Mat_Close(matfp);
        free(rowptr); free(index);
        return 1;
    }

    for (int i = 0; i < nrows; i++) {
        temp_rowptr[i] = rowptr[i];
    }

    for (int j = 0; j < ncols; j++) {
        for (int p = jc_original[j]; p < jc_original[j+1]; p++) {
            int i = ir_original[p]; 
            if (i < nrows) {
                int dest = temp_rowptr[i]++;
                index[dest] = j; 
            }
        }
    }

    Mat_Close(matfp);
    free(temp_rowptr);
    Mat_VarFree(top_level_var);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    int *labels = malloc(nrows*sizeof(int));
    if (!labels) {
        printf("Memory allocation failed\n");
        free(rowptr); free(index);
        MPI_Finalize();
        return 1;
    }

    gettimeofday(&start, NULL);
    coloringCC_mpi_openmp(nrows, rowptr, index, labels, rank, size, MPI_COMM_WORLD);
    gettimeofday(&end, NULL);

    if (rank==0){
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6;
        printf("Execution time: %f seconds\n", elapsed);
        fflush(stdout);

        /*printf("Global Labels Vector Assembled on Rank 0:\n");
        fflush(stdout);
        for (int i=0; i<nrows; i++){
            printf("%d ", labels[i]);
        }*/
    }
    free(index);free(rowptr);
    free(labels);
    MPI_Finalize();
}

