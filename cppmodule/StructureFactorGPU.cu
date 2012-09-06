/*! \file StructureFactorGPU.cu
    \brief CUDA implementation of StructureFactor GPU routines
 */
#include <cuda.h>

#include "StructureFactorGPU.cuh"

__global__ void kernel_calculate_sq_partial(
            int n_particles,
            Scalar2 *fourier_mode_partial,
            Scalar4 *postype,
            int n_wave,
            Scalar3 *wave_vectors,
            Scalar *d_modes)
    {
    extern __shared__ Scalar2 sdata[];

    unsigned int tidx = threadIdx.x;

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = 0; i < n_wave; i++) {
        Scalar3 q = wave_vectors[i];

        Scalar2 mySum = make_scalar2(0.0,0.0);

        if (j < n_particles) {

            Scalar3 p = make_scalar3(postype[j].x, postype[j].y, postype[j].z);
            Scalar dotproduct = q.x * p.x + q.y * p.y + q.z * p.z;
            unsigned int type = __float_as_int(postype[j].w);
            Scalar mode = d_modes[type];
            mySum.x += mySum.x+mode*cosf(dotproduct);
            mySum.y += mySum.y+mode*sinf(dotproduct);
        }
        sdata[tidx] = mySum;

       __syncthreads();
        // reduce in shared memory
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (tidx < offs)
                {
                mySum.x += sdata[tidx+offs].x;
                mySum.y += sdata[tidx+offs].y;
                sdata[tidx] = mySum;
                }
            offs >>=1;
            __syncthreads();
            }

        // write result to global memeory
        if (tidx == 0)
           fourier_mode_partial[blockIdx.x + gridDim.x*i] = sdata[0];
        } // end loop over wave vectors
    } 

__global__ void kernel_final_reduce_fourier_modes(Scalar2* fourier_mode_partial,
                                       unsigned int nblocks,
                                       Scalar2 *fourier_modes,
                                       unsigned int n_wave)
    {
    extern __shared__ Scalar2 smem[];

    unsigned int j = blockIdx.x;

    if (threadIdx.x == 0)
       fourier_modes[j] = make_scalar2(0.0,0.0);

    for (int start = 0; start< nblocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < nblocks)
            smem[threadIdx.x] = fourier_mode_partial[j*nblocks+start + threadIdx.x];
        else
            smem[threadIdx.x] = make_scalar2(0.0,0.0);

        __syncthreads();

        // reduce the sum
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                {
                smem[threadIdx.x].x += smem[threadIdx.x + offs].x;
                smem[threadIdx.x].y += smem[threadIdx.x + offs].y;
                }
            offs >>= 1;
            __syncthreads();
            }

         if (threadIdx.x == 0)
            {
            fourier_modes[j].x += smem[0].x;
            fourier_modes[j].y += smem[0].y;
            }
        }
    }

cudaError_t gpu_calculate_fourier_modes(unsigned int n_wave,
                                 Scalar3 *d_wave_vectors,
                                 unsigned int n_particles,
                                 Scalar4 *d_postype,
                                 Scalar *d_mode,
                                 Scalar2 *d_fourier_modes,
                                 unsigned int block_size,
                                 Scalar2 *d_fourier_mode_partial
                                 )
    {
    cudaError_t cudaStatus;

    unsigned int n_blocks = n_particles/block_size + 1;

    unsigned int shared_size = block_size * sizeof(Scalar2);
    kernel_calculate_sq_partial<<<n_blocks, block_size, shared_size>>>(
               n_particles,
               d_fourier_mode_partial,
               d_postype,
               n_wave,
               d_wave_vectors,
               d_mode);

    if (cudaStatus = cudaGetLastError()) 
           return cudaStatus;

    // calculate final S(q) values 
    const unsigned int final_block_size = 512;
    shared_size = final_block_size*sizeof(Scalar2);
    kernel_final_reduce_fourier_modes<<<n_wave, final_block_size,shared_size>>>(d_fourier_mode_partial,
                                                                  n_blocks,
                                                                  d_fourier_modes,
                                                                  n_wave);
                                                                  

    if (cudaStatus = cudaGetLastError())
        return cudaStatus;

    return cudaSuccess;
    }
