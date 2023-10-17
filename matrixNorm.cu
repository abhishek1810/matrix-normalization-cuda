/* Matrix normalization.
 * Compile with "gcc matrixNorm.c"
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

/* Program Parameters */
#define N 12000  /* Matrix size */

int numBlocks = 32;
int numThreadsPerBlock = 64;

/* Matrices */
volatile float A[N][N], B_cpu[N][N], B_gpu[N][N];

/* Initialize A and B*/
void initialize_inputs() {
    int row, col;
    
    srand((unsigned)time(NULL));
    // srand(0);
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[row][col] = (float)rand() / 32768.0;
            B_cpu[row][col] = 0.0;
            B_gpu[row][col] = 0.0;
        }
    }   
}

/* Serial function */
void matrixNormSerially() {
    int row, col;
    float mu, sigma; // Mean and Standard Deviation
    
    printf("Computing Serially.\n");
    
    for (col=0; col < N; col++) {
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += A[row][col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(A[row][col] - mu, 2.0);
        sigma /= (float) N;
        sigma = sqrt(sigma);
        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                B_cpu[row][col] = 0.0;
            else
                B_cpu[row][col] = (A[row][col] - mu) / sigma;
        }
    }
}

/* Method to check correctness of parallel program*/
void checkCorrectness() {
    float sum_cpu = 0; 
    float sum_gpu = 0;
    int row, col;
    for (row=0; row < N; row++) {
        for (col=0; col < N; col++) {
            sum_cpu += B_cpu[row][col];
            sum_gpu += B_gpu[row][col];
        }
    }
    printf("Sum of normalized array by CPU : %f\n", sum_cpu);
    printf("Sum of normalized array by GPU : %f\n", sum_gpu);
}

/* Kernel function */
__global__ void matrixNorm (float *d_A, float *d_B, int n, int totalThreads) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row;
    float mu, sigma;

    for (;col < n; col += totalThreads){
        mu = (float)0.0;
        for (row=0; row < n; row++)
            mu += d_A[row*n+col];
        mu /= (float) n;
        
        // __syncthreads();
        
        sigma = (float)0.0;
        for (row=0; row < n; row++)
            sigma += powf(d_A[row*n+col] - mu, (float)2.0);
        sigma /= (float) n;

        // __syncthreads();
        sigma = sqrt( (float) sigma);

        for (row=0; row < n; row++) {
            if (sigma == (float)0.0)
                d_B[row*n+col] = (float)0.0;
            else
                d_B[row*n+col] = (d_A[row*n+col] - mu) / sigma;
        }
    }
}

/* Print input matrices */
void print_matrix() {
  int row, col;

  if (N <= 5) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB_cpu = [");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	    printf("%5.2f%s", B_cpu[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB_gpu = [");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	    printf("%5.2f%s", B_gpu[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\n");
  }
}

int main(int argc, char **argv) {
    /* Timing variables */
    struct timeval start, stop;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    unsigned long long runtime;

    /* Initialize A and B */
    initialize_inputs();

    printf("Computing in Parallel\n");

    float *d_A, *d_B;

    /* Start Clock */
    printf("\n---------------------------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");
    gettimeofday(&start, &tzdummy);

    cudaMalloc((void **) &d_A, sizeof(float)*N*N);
    cudaMalloc((void **) &d_B, sizeof(float)*N*N);

    cudaMemcpy(d_A, (const void *)A, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    matrixNorm<<<numBlocks,numThreadsPerBlock>>>(d_A, d_B, N, numBlocks*numThreadsPerBlock);

    cudaMemcpy((void *)B_gpu, (d_B), sizeof(float)*N*N, cudaMemcpyDeviceToHost);

    /* Stop Clock */
    gettimeofday(&stop, &tzdummy);
    runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);

    /* Display output */
    // print_B();

    cudaFree(d_A);
    cudaFree(d_B);

    /* Display timing results */
    printf("Runtime = %g ms.\n", (float)runtime/(float)1000);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");
    
    printf("Computing in Serial\n");
    matrixNormSerially();

    print_matrix();

    printf("Checking correctness\n");
    checkCorrectness();

    exit(0);
}
