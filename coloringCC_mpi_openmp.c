#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

#include "coloringCC_mpi_openmp.h"

/*void coloringCC_mpi_openmp(int nrows, const int *rowptr, const int *index, int *labels, int rank, int size, MPI_Comm comm) {

    int chunk_size = (nrows+size-1)/size;
    int start_ = rank*chunk_size;
    int end_ = (start_+chunk_size>nrows) ? nrows : start_+chunk_size;

    int *reduced = malloc(nrows*sizeof(int));
    if (!reduced) {
        printf("Memory allocation failed\n");
        return;
    }

    int *old_labels = malloc(nrows * sizeof(int));
    int *new_labels = malloc(nrows * sizeof(int));
    if (!old_labels || !new_labels) {
        free(old_labels); free(new_labels);
        return;
    }   

    omp_set_num_threads(NUM_OF_THREADS);

    #pragma omp parallel for schedule(static)
    for (int i=0; i<nrows; i++) {
        old_labels[i] = i;
        new_labels[i] = i;
    }

    int global_changed = 1;
   
    while (global_changed) {
        
        int local_changed = 0;

        #pragma omp parallel for schedule(static) //schedule(dynamic, ...)
        for (int v=start_; v<end_; v++){
            int start = rowptr[v];
            int end = rowptr[v+1];
            int lv = old_labels[v];
            for (int j=start; j<end; j++) {
                int u = index[j];
                int lu = old_labels[u];
                if (lu<lv) lv = lu;
            }

            new_labels[v] = lv;

            if (lv != old_labels[v]) local_changed = 1;
        }
        int *tmp = old_labels;
        old_labels = new_labels;
        new_labels = tmp;

        MPI_Allreduce(old_labels, reduced, nrows, MPI_INT, MPI_MIN, comm);

        memcpy(old_labels, reduced, nrows * sizeof(int));

        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, comm);
    }
    memcpy(labels, old_labels, nrows * sizeof(int));
    free(old_labels); free(new_labels);free(reduced);
}*/

void coloringCC_mpi_openmp(int nrows, const int *rowptr, const int *index, int *labels, int rank, int size, MPI_Comm comm) {

    int chunk_size = (nrows+size-1)/size;
    int start_ = rank*chunk_size;
    int end_ = (start_+chunk_size>nrows) ? nrows : start_+chunk_size;

    int *old_labels = malloc(nrows*sizeof(int));
    int *new_labels = malloc(nrows*sizeof(int));
    if (!old_labels || !new_labels) {
        free(old_labels); free(new_labels);
        return;
    }   

    #pragma omp parallel for schedule(static)
    for (int i=start_; i<end_; i++) {
        old_labels[i] = i;
        new_labels[i] = i;
    }

    int **ghost_lists = calloc(size, sizeof(int*));
    int *ghost_counts = calloc(size, sizeof(int));
    int *ghost_capacities = calloc(size, sizeof(int));
    if (!ghost_lists || !ghost_counts || !ghost_capacities) {
        free(ghost_lists); free(ghost_counts); free(ghost_capacities);
        free(old_labels); free(new_labels);
        return;
    }

    char *seen = calloc(nrows, sizeof(char));

    for (int v=start_; v<end_; v++) {
        for (int j=rowptr[v]; j<rowptr[v+1]; j++) {
            int u = index[j];
            int owner;
            if (u >= (size - 1) * chunk_size) {
                owner = size - 1;
            } else {
                owner = u / chunk_size;
            }
            if (owner!=rank && !seen[u]) {
                seen[u] = 1;

                if (ghost_counts[owner] == ghost_capacities[owner]) {
                    ghost_capacities[owner] = (ghost_capacities[owner]==0) ? 1 : ghost_capacities[owner]*2;
                    ghost_lists[owner] = realloc(ghost_lists[owner], ghost_capacities[owner]*sizeof(int));
                    if (!ghost_lists[owner]) {
                        for (int r=0; r<size; r++) {
                            free(ghost_lists[r]);
                        }
                        free(ghost_lists);
                        free(ghost_counts);
                        free(ghost_capacities);
                        free(old_labels); free(new_labels);
                        free(seen);
                        return;
                    }
                }
                ghost_lists[owner][ghost_counts[owner]++] = u;
            }
        }
          
    }  
    free(seen);

    int *send_counts = calloc(size, sizeof(int));
    if (!send_counts) {
        for (int r=0; r<size; r++) {
            free(ghost_lists[r]);
        }
        free(ghost_lists);
        free(ghost_counts);
        free(ghost_capacities);
        free(old_labels); free(new_labels);
        return;
    }

    MPI_Alltoall(ghost_counts, 1, MPI_INT, send_counts, 1, MPI_INT, comm);

    int *send_displs = calloc(size, sizeof(int));
    int *recv_displs = calloc(size, sizeof(int));
    if (!send_displs || !recv_displs) {
        for (int r=0; r<size; r++) {
            free(ghost_lists[r]);
        }
        free(ghost_lists);
        free(ghost_counts);
        free(ghost_capacities);
        free(send_counts);
        free(send_displs);
        free(recv_displs);
        free(old_labels); free(new_labels);
        return;
    }

    for (int r=1; r<size; r++) {
        send_displs[r] = send_displs[r-1] + send_counts[r-1];
        recv_displs[r] = recv_displs[r-1] + ghost_counts[r-1];
    }

    int total_send = (int)(send_displs[size-1] + send_counts[size-1]);
    int total_recv = (int)(recv_displs[size-1] + ghost_counts[size-1]);

    int *send_vertices = malloc(total_send*sizeof(int));
    int *recv_vertices = malloc(total_recv*sizeof(int));
    if (!send_vertices || !recv_vertices) {
        for (int r=0; r<size; r++) {
            free(ghost_lists[r]);
        }
        free(ghost_lists);
        free(ghost_counts);
        free(ghost_capacities);
        free(send_counts);
        free(send_displs);
        free(recv_displs);
        free(send_vertices);
        free(recv_vertices);
        free(old_labels); free(new_labels);
        return;
    }

    for (int r=0; r<size; r++) {
        if (ghost_counts[r]>0) {
            memcpy(recv_vertices + recv_displs[r], ghost_lists[r], ghost_counts[r]*sizeof(int));
        }
    }

    MPI_Alltoallv(recv_vertices, ghost_counts, recv_displs, MPI_INT, send_vertices, send_counts, send_displs, MPI_INT, comm);

    int *send_labels = malloc(total_send * sizeof(int));
    int *recv_labels = malloc(total_recv * sizeof(int));
    if (!send_labels || !recv_labels) {
        for (int r=0; r<size; r++) {
            free(ghost_lists[r]);
        }
        free(ghost_lists);
        free(ghost_counts);
        free(ghost_capacities);
        free(send_counts);
        free(send_displs);
        free(recv_displs);
        free(send_vertices);
        free(recv_vertices);
        free(send_labels);
        free(recv_labels);
        free(old_labels); free(new_labels);
        return;
    }

    for (int i=0; i<total_send; i++) {
            send_labels[i] = old_labels[send_vertices[i]];
        }

    MPI_Alltoallv(send_labels, send_counts, send_displs, MPI_INT, recv_labels, ghost_counts, recv_displs, MPI_INT, comm);

    for (int i=0; i<total_recv; i++) {
            old_labels[recv_vertices[i]] = recv_labels[i];
        }
    
    int global_changed = 1;
    int local_changed;
    int K = 6;
   
    while (global_changed) {
        
        local_changed = 0;

        for (int iter=0; iter<K; iter++) {
            #pragma omp parallel for schedule(static)//schedule(dynamic, 1024)
            for (int v=start_; v<end_; v++){
                int start = rowptr[v];
                int end = rowptr[v+1];
                int lv = old_labels[v];
                for (int j=start; j<end; j++) {
                    int u = index[j];
                    int lu = old_labels[u];
                    if (lu<lv) lv = lu;
                }

                new_labels[v] = lv;

                if (lv != old_labels[v]) local_changed = 1;
            }
            int *tmp = old_labels;
            old_labels = new_labels;
            new_labels = tmp;
        }

        for (int i=0; i<total_send; i++) {
            send_labels[i] = old_labels[send_vertices[i]];
        }

        MPI_Alltoallv(send_labels, send_counts, send_displs, MPI_INT, recv_labels, ghost_counts, recv_displs, MPI_INT, comm);

        for (int i=0; i<total_recv; i++) {
            int vertex = recv_vertices[i];
            if (recv_labels[i] < old_labels[vertex]) {
                old_labels[vertex] = recv_labels[i];
                local_changed = 1;
            }
        }

        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, comm);
    }

    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank==0) {
        recvcounts = malloc(size*sizeof(int));
        displs = malloc(size*sizeof(int));
        if (!recvcounts || !displs) {
            for (int r=0; r<size; r++) {
                free(ghost_lists[r]);
            }
            free(ghost_lists);
            free(ghost_counts);
            free(ghost_capacities);
            free(send_counts);
            free(send_displs);
            free(recv_displs);
            free(send_vertices);
            free(recv_vertices);
            free(send_labels);
            free(recv_labels);
            free(old_labels); free(new_labels);
            free(recvcounts);
            free(displs);
            return;
        }

        for (int r=0; r<size; r++) {
            int r_start = (r*chunk_size>nrows) ? nrows : (int)(r*chunk_size);
            int r_end = (r_start + chunk_size>nrows) ? nrows : r_start + chunk_size;
            recvcounts[r] = (int)(r_end - r_start);
            displs[r] = (int)r_start;
        }
    }

    int local_size = end_-start_;
    MPI_Gatherv(old_labels + start_, local_size, MPI_INT, labels, recvcounts, displs, MPI_INT, 0, comm);
    
    for (int r=0; r<size; r++) {
        free(ghost_lists[r]);
    }
    free(ghost_lists);
    free(ghost_counts);
    free(ghost_capacities);
    free(send_counts);
    free(send_displs);
    free(recv_displs);
    free(send_vertices);
    free(recv_vertices);
    free(send_labels);
    free(recv_labels);
    free(old_labels);
    free(new_labels);
    if (rank==0) {
        free(recvcounts);
        free(displs);
    }
}
