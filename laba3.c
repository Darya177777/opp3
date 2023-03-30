#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<time.h>
#include<sys/time.h>

#define NUM_DIMS 2
#define XLATTICE 2
#define YLATTICE 2
#define M 2000
#define N 2000
#define K 2000

void InitMatrix(double **MatrixA, double **MatrixB, double **MatrixC) {
    *MatrixA = (double *)malloc(M * N * sizeof(double));
    *MatrixB = (double *)malloc(N * K * sizeof(double));
    *MatrixC = (double *)malloc(M * K * sizeof(double));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            (*MatrixA)[N * i + j] = 1;
    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++)
            (*MatrixB)[K * j + k] = 1;
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            (*MatrixC)[K * i + k] = 0.0;
}

void SimpleMultiply(double **SubMatrixA, double **SubMatrixB, double **SubMatrixC, int SizeBandA, int SizeBandB) {
    for (int i = 0; i < SizeBandA; i++) {
        for (int j = 0; j < SizeBandB; j++) {
            (*SubMatrixC)[SizeBandB * i + j] = 0.0;
            for (int k = 0; k < N; k++) {
                (*SubMatrixC)[SizeBandB * i + j] = (*SubMatrixC)[SizeBandB * i + j] + (*SubMatrixA)[N * i + k] * (*SubMatrixB)[SizeBandB * k + j];
            }
        }
    }
}

void FreeData(double *SubMatrixA, double *SubMatrixB, double *SubMatrixC, int rank, MPI_Comm *pcomm, MPI_Comm *comm2D,
              MPI_Comm *comm1D, int * countc, int * dispc, MPI_Datatype typeb, MPI_Datatype typec, MPI_Datatype *types){
    if (SubMatrixA != NULL && SubMatrixB != NULL && SubMatrixC != NULL){
        free(SubMatrixA);
        free(SubMatrixB);
        free(SubMatrixC);
    }
    MPI_Comm_free(pcomm);
    MPI_Comm_free(comm2D);
    for (int i = 0; i < 2; i++) {
        MPI_Comm_free(&comm1D[i]);
    }
    if (rank == 0) {
        free(countc);
        free(dispc);
        MPI_Type_free(&typeb);
        MPI_Type_free(&typec);
        MPI_Type_free(&types[0]);
    }
}

int MultiplyMatrix(double **MatrixA, double **MatrixB, double **MatrixC, MPI_Comm comm) {
    int SizeBandA = M / XLATTICE;
    int SizeBandB = K / YLATTICE;
    int coords[NUM_DIMS];
    int rank;
    int *countc = NULL, *dispc = NULL, *countb = NULL, *dispb = NULL;
    MPI_Datatype typeb, typec, types[NUM_DIMS];
    int blockLen[NUM_DIMS] = {1, 1};
    int dims[NUM_DIMS] = {XLATTICE, YLATTICE};
    int periods[NUM_DIMS] = {0, 0};
    int remainDims[NUM_DIMS];
    MPI_Aint disp[NUM_DIMS];
    MPI_Comm comm2D, comm1D[NUM_DIMS], pcomm;
    MPI_Comm_dup(comm, &pcomm);
    MPI_Cart_create(pcomm, NUM_DIMS, dims, periods, 0, &comm2D);
    MPI_Comm_rank(comm2D, &rank);
    MPI_Cart_coords(comm2D, rank, NUM_DIMS, coords);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++)
            if (i == j)
                remainDims[j] = 1;
            else
                remainDims[j] = 0;
        MPI_Cart_sub(comm2D, remainDims, &comm1D[i]);
    }
    double * SubMatrixA = (double *)malloc(SizeBandA * N * sizeof(double));
    double * SubMatrixB = (double *)malloc(N * SizeBandB * sizeof(double));
    double * SubMatrixC = (double *)malloc(SizeBandA * SizeBandB * sizeof(double));

    if (rank == 0) {
        // для передачи столбца B
        MPI_Type_vector(N, SizeBandB, K, MPI_DOUBLE, &types[0]);
        disp[0] = 0;
        disp[1] = sizeof(double) * SizeBandB; // смещение
        types[1] = MPI_UB;
        MPI_Type_struct(2, blockLen, disp, types, &typeb);
        MPI_Type_commit(&typeb);
        // для смещения подматрицы B в матрице B
        dispb = (int *) malloc(YLATTICE * sizeof(int));
        countb = (int *) malloc(YLATTICE * sizeof(int));
        for (int i = 0; i < YLATTICE; i++) {
            dispb[i] = i;
            countb[i] = 1;
        }
        // для подматрицы C
        MPI_Type_vector(SizeBandA, SizeBandB, K, MPI_DOUBLE, &types[0]);
        MPI_Type_struct(2, blockLen, disp, types, &typec);
        MPI_Type_commit(&typec);
        dispc = (int *) malloc(XLATTICE * YLATTICE * sizeof(int));
        countc = (int *) malloc(XLATTICE * YLATTICE * sizeof(int));
        for (int i = 0; i < XLATTICE; i++) {
            for (int j = 0; j < YLATTICE; j++) {
                dispc[i * YLATTICE + j] = (i * YLATTICE * SizeBandA + j);
                countc[i * YLATTICE + j] = 1;
            }
        }
    }
    if (coords[1] == 0)
        MPI_Scatter(*MatrixA, SizeBandA * N, MPI_DOUBLE, SubMatrixA, SizeBandA * N, MPI_DOUBLE, 0, comm1D[0]);
    if (coords[0] == 0)
        MPI_Scatterv(*MatrixB, countb, dispb, typeb, SubMatrixB, N * SizeBandB, MPI_DOUBLE, 0, comm1D[1]);

    MPI_Bcast(SubMatrixA, SizeBandA * N, MPI_DOUBLE, 0, comm1D[1]);
    MPI_Bcast(SubMatrixB, N * SizeBandB, MPI_DOUBLE, 0, comm1D[0]);
    SimpleMultiply(&SubMatrixA, &SubMatrixB, &SubMatrixC, SizeBandA, SizeBandB);
    if (SubMatrixC != NULL)
        MPI_Gatherv(SubMatrixC, SizeBandA * SizeBandB, MPI_DOUBLE, *MatrixC, countc, dispc, typec, 0, comm2D);
    MPI_Comm * ptr = comm1D;
    FreeData(SubMatrixA, SubMatrixB, SubMatrixC, rank, &pcomm, &comm2D, ptr, countc, dispc, typeb, typec, types);
    return 0;
}


int main(int argc, char **argv) {
    int size, process;
    int dims[NUM_DIMS], periods[NUM_DIMS];
    double *MatrixA = NULL, *MatrixB = NULL, *MatrixC = NULL;
    MPI_Comm comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &process);
    for (int i = 0; i < NUM_DIMS; i++) {
        dims[i] = 0;
        periods[i] = 0;
    }
    MPI_Dims_create(size, NUM_DIMS, dims);
    MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, 0, &comm);
    if (process == 0)
        InitMatrix(&MatrixA, &MatrixB, &MatrixC);
    double t = MPI_Wtime();
    MultiplyMatrix(&MatrixA, &MatrixB, &MatrixC, comm);
    printf("Number of process = %d Time = %10.5f\n", process, MPI_Wtime() - t);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    if (process == 0) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++)
                printf(" %3.1f", MatrixC[K * i + j]);
            printf("\n");
        }
        if (MatrixA != NULL && MatrixB != NULL && MatrixC != NULL) {
            free(MatrixA);
            free(MatrixB);
            free(MatrixC);
        }
    }
    return 0;
}
