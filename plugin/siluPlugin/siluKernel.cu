/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#if CUDA_VERSION >= 10010

#include <cassert>
#include <cstring>
#include <vector>

#include "siluPlugin.h"

using namespace nvinfer1;

__device__ __forceinline__ float sigmoid (float a)
{
    return 1.0 / (1.0 + exp (-a));
}

void __global__ silu_kernel(int n, float* inputData, float* outputData) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        inputData[tid] = outputData[tid] * sigmoid(outputData[tid]);
    }
}

int SiLUInference(cudaStream_t stream, int size, float* inputData, float* outputData){
    const int nThreads = 512;

    int nBlocks = (size + nThreads - 1) / nThreads;
    
    silu_kernel<<<nThreads, nBlocks>>>(size, inputData, outputData);

    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
             __FILE__, __LINE__, cudaGetErrorString( err ) );
        return 1;
    }
    return 0;
}

#endif // CUDA_VERSION >= 10010
