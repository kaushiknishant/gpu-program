#include <iostream>
#include <cuda_runtime.h>


__host__ __device__ float f(float a, float b){
    return a + b;
}

void vecadd_cpu(float* x, float* y, float* z, int N){
    for(unsigned int i = 0; i < N; ++i){
        z[i] = f(x[i], y[i]);
    }
}

__global__ void vecadd_kernel(float* x, float* y, float* z, int N){
    // Grid Dimension ---> gridDim.x tells no. of blocks in the grid
    // Block Index ---> blockIdx.x  tells index of its block w.r.t other blocks in grid
    // Block dimension ---> blockDim.x tells threads in block(size of block)
    // Thread Index ---> threadIdx.x tells position of thread in block
    unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
    if(i < N){
        z[i] = f(x[i], y[i]);
    }

    // single program multiple data 

}

void vecadd_gpu(float* x, float* y, float* z, int N){
    // Allocate GPU memory
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, N*sizeof(float));
    cudaMalloc((void**)&y_d, N*sizeof(float));
    cudaMalloc((void**)&z_d, N*sizeof(float));

    // Copy to the GPU
    cudaMemcpy(x_d, x, N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float),cudaMemcpyHostToDevice);

    // Run the GPU code
    // call a GPU kernel function(launch a grid of threads)
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1)/512;
    vecadd_kernel<<< numBlocks, numThreadsPerBlock >>>(x_d, y_d, z_d, N);

    // Copy from the GPU
    cudaMemcpy(z, z_d, N*sizeof(float),cudaMemcpyDeviceToHost);

    // Deallocate GPu memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}


int main(int argc, char**argv) {

    cudaDeviceSynchronize();
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Allocate memory and initialize data
    unsigned int N = (argc > 1)?(atoi(argv[1])): (1 << 25) ;
    float* x = (float*) malloc(N*sizeof(float));
    float* y = (float*) malloc(N*sizeof(float));
    float* z = (float*) malloc(N*sizeof(float));
    for (unsigned int i = 0; i < N; ++i) {
        x[i] = rand();
        y[i] = rand();
    }

    //vector addition on CPU
    cudaEventRecord(start);
    vecadd_cpu(x,y,z,N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "CPU program took " << milliseconds << " milliseconds to execute." << std::endl;

    //vector addition on GPU
    cudaEventRecord(start);
    vecadd_gpu(x,y,z,N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "GPU program took " << milliseconds << " milliseconds to execute." << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}