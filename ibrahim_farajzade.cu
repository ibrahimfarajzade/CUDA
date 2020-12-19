/*

 Ibrahim Farajzade - CUDA Project
 Matrix Operations

 Output:
 Please type in m, n and k: 128 256 512
 Time elapsed on matrix multiplication of 128x256 . 256x512 on GPU: 6.29261ms.
 Time elapsed on matrix multiplication of 128x256 . 256x512 on CPU: 0.002048ms.
 All results are correct, speedup = 0.000325461

*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
using namespace std;

#define BLOCK_SIZE 16

// Dot product of two matrix 
__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	if (col < k && row < m) {
		for (int i = 0; i < n; i++) {
			sum += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = sum;
	}
}

// Dot production of two matrix in GPU
__global__ void gpu_square_matrix_mult(int *d_a, int *d_b, int *d_result, int n) {
	__shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int tmp = 0;
	int idx;

	for (int sub = 0; sub < gridDim.x; ++sub) {
		idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
		if (idx >= n * n) {
			// n may not divisible by BLOCK_SIZE
			tile_a[threadIdx.y][threadIdx.x] = 0;
		}
		else {
			tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
		}

		idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
		if (idx >= n * n) {
			tile_b[threadIdx.y][threadIdx.x] = 0;
		} else {
			tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
		}
		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k) {
			tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
		}
		__syncthreads();
	}
	if (row < n && col < n) {
		d_result[row * n + col] = tmp;
	}
}

// Matrix transpose
__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < cols && idy < rows) {
		unsigned int pos = idy * cols + idx;
		unsigned int trans_pos = idx * rows + idy;
		mat_out[trans_pos] = mat_in[pos];
	}
}

// Dot product of two matrix in CPU, for validationg GPU results
void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			int tmp = 0.0;
			for (int h = 0; h < n; ++h) {
				tmp += h_a[i * n + h] * h_b[h * k + j];
			}
			h_result[i * k + j] = tmp;
		}
	}
}

int main(int argc, char const *argv[]) {
	int m, n, k;
	/* Fixed seed for illustration */
	srand(3333);
	cout << "Please type in m, n and k: ";
	cin >> m >> n >> k;

	// Allocate memory in host RAM, h_cc is used to store CPU result
	int *h_a, *h_b, *h_c, *h_cc;
	cudaMallocHost((void **)&h_a, sizeof(int)*m*n);
	cudaMallocHost((void **)&h_b, sizeof(int)*n*k);
	cudaMallocHost((void **)&h_c, sizeof(int)*m*k);
	cudaMallocHost((void **)&h_cc, sizeof(int)*m*k);

	// Random initialize matrix A
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			h_a[i * n + j] = rand() % 1024;
		}
	}

	// Random initialize matrix B
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < k; ++j) {
			h_b[i * k + j] = rand() % 1024;
		}
	}

	float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

	// Some events to count the execution time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start to count execution time of GPU version
	cudaEventRecord(start, 0);

	// Allocate memory space on the device 
	int *d_a, *d_b, *d_c;
	cudaMalloc((void **)&d_a, sizeof(int)*m*n);
	cudaMalloc((void **)&d_b, sizeof(int)*n*k);
	cudaMalloc((void **)&d_c, sizeof(int)*m*k);

	// Copy matrix A and B from host to device memory
	cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

	unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	// Launch kernel 
	if (m == n && n == k) {
		gpu_square_matrix_mult << <dimGrid, dimBlock >> > (d_a, d_b, d_c, n);
	} else {
		gpu_matrix_mult << <dimGrid, dimBlock >> > (d_a, d_b, d_c, m, n, k);
	}

	// Transefr results from device to host 
	cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	// Time counting terminate
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Compute time elapse on GPU computing
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	cout << "Time elapsed on matrix multiplication of " << m << "x" << n << " . " << n << "x" << k << " on GPU: " << gpu_elapsed_time_ms << "ms." << endl;

	// Start the CPU version
	cudaEventRecord(start, 0);

	cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
	cout << "Time elapsed on matrix multiplication of " << m << "x" << n << " . " << n << "x" << k << " on CPU: " << cpu_elapsed_time_ms << "ms." << endl;

	// Validate results computed by GPU
	int all_ok = 1;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			if (h_cc[i*k + j] != h_c[i*k + j]) {
				all_ok = 0;
			}
		}
	}

	// Roughly compute speedup
	if (all_ok) {
		cout << "All results are correct, speedup = " << cpu_elapsed_time_ms / gpu_elapsed_time_ms << endl;
	} else {
		cout << "Incorrect results" << endl;
	}

	// Free memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);
	cudaFreeHost(h_cc);

	return 0;
}