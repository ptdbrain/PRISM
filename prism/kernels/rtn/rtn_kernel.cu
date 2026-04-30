#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700) || defined(USE_ROCM)
// adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh

__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
    unsigned int *address_as_ui = reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) - (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short hsum = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
        hsum += val;
        old = reinterpret_cast<size_t>(address) & 2
                 ? (old & 0xffff) | (hsum << 16)
                 : (old & 0xffff0000) | hsum;
        old = atomicCAS(address_as_ui, assumed, old);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
}
__device__ __forceinline__ void atomicAdd(__half* address, c10::Half val) {
    unsigned int * address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        __half_raw hsum;
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        half tmpres = __hadd(hsum, val);
        hsum = __half_raw(tmpres);
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}
#endif

__global__ void VecQuant2MatMulKernelRTN(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    int batch,
    int vec_height,
    int height,
    int width,
    int groupsize
);

__global__ void VecQuant3MatMulKernelRTN(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    int batch,
    int vec_height,
    int height,
    int width,
    int groupsize
);

__global__ void VecQuant4MatMulKernelRTN(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    int batch,
    int vec_height,
    int height,
    int width,
    int groupsize
);

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT2 =  16;
const int BLOCKHEIGHT3 =  24;
const int BLOCKHEIGHT4 =  32;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}


void vecquant2matmul_rtn_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant2MatMulKernelRTN<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    batch, vec_height, height, width, groupsize
  );
}

__global__ void VecQuant2MatMulKernelRTN(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    int batch,
    int vec_height,
    int height,
    int width,
    int groupsize
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT2 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[16][16];
  int val = threadIdx.x / 16;
  int off = threadIdx.x % 16;
  for (; val < 16; val += BLOCKWIDTH / 16) {
    // 2-bit symmetric: qmax = 1
    deq2[val][off] = __halves2half2(
       __int2half_rn((val & 0x3) - 1), __int2half_rn((val >> 2) - 1)
    );
  }

  int i = width * h + w;
  int g_h = h * 16;
  int k = 0;

  float res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
    float scale_f = scales[g * width + w];
    half2 scale = __float2half2_rn(scale_f);

    std::memset(&res2, 0, sizeof(half2));
    tmp = as_unsigned(mat[i]);
    res2 = __hfma2(__hmul2(deq2[(tmp >>  0) & 0xf][off], scale), blockvec[k + 0], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp >>  4) & 0xf][off], scale), blockvec[k + 1], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp >>  8) & 0xf][off], scale), blockvec[k + 2], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp >> 12) & 0xf][off], scale), blockvec[k + 3], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp >> 16) & 0xf][off], scale), blockvec[k + 4], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp >> 20) & 0xf][off], scale), blockvec[k + 5], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp >> 24) & 0xf][off], scale), blockvec[k + 6], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp >> 28) & 0xf][off], scale), blockvec[k + 7], res2);
	i += width;
    k += 8;
    res += __low2float(res2) + __high2float(res2);
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant3matmul_rtn_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelRTN<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    batch, vec_height, height, width, groupsize
  );
}

__global__ void VecQuant3MatMulKernelRTN(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    int batch,
    int vec_height,
    int height,
    int width,
    int groupsize
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT3 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[64][32];
  int val = threadIdx.x / 32;
  int off = threadIdx.x % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    // 3-bit symmetric: qmax = 3
    deq2[val][off] = __halves2half2(
       __int2half_rn((val & 0x7) - 3), __int2half_rn((val >> 3) - 3)
    );
  }

  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k = 0;

  float res = 0;
  half2 res2;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
	float scale_f = scales[g * width + w];
    half2 scale = __float2half2_rn(scale_f);

    std::memset(&res2, 0, sizeof(half2));
    tmp1 = as_unsigned(mat[i]);
    res2 = __hfma2(__hmul2(deq2[(tmp1 >>  0) & 0x3f][off], scale), blockvec[k + 0], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp1 >>  6) & 0x3f][off], scale), blockvec[k + 1], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp1 >> 12) & 0x3f][off], scale), blockvec[k + 2], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp1 >> 18) & 0x3f][off], scale), blockvec[k + 3], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp1 >> 24) & 0x3f][off], scale), blockvec[k + 4], res2);
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
    res2 = __hfma2(__hmul2(deq2[tmp][off], scale), blockvec[k + 5], res2);
    tmp2 >>= 4;
    k += 6;
    res2 = __hfma2(__hmul2(deq2[(tmp2 >>  0) & 0x3f][off], scale), blockvec[k + 0], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp2 >>  6) & 0x3f][off], scale), blockvec[k + 1], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp2 >> 12) & 0x3f][off], scale), blockvec[k + 2], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp2 >> 18) & 0x3f][off], scale), blockvec[k + 3], res2);
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
    res2 = __hfma2(__hmul2(deq2[tmp][off], scale), blockvec[k + 4], res2);
    tmp1 >>= 2;
    k += 5;
    res2 = __hfma2(__hmul2(deq2[(tmp1 >>  0) & 0x3f][off], scale), blockvec[k + 0], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp1 >>  6) & 0x3f][off], scale), blockvec[k + 1], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp1 >> 12) & 0x3f][off], scale), blockvec[k + 2], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp1 >> 18) & 0x3f][off], scale), blockvec[k + 3], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp1 >> 24) & 0x3f][off], scale), blockvec[k + 4], res2);
    i += width;
    k += 5;
    res += __low2float(res2) + __high2float(res2);
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant4matmul_rtn_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelRTN<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    batch, vec_height, height, width, groupsize
  );
}

__global__ void VecQuant4MatMulKernelRTN(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    int batch,
    int vec_height,
    int height,
    int width,
    int groupsize
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[256][8];
  int val = threadIdx.x / 8;
  int off = threadIdx.x % 8;
  for (; val < 256; val += BLOCKWIDTH / 8) {
    // 4-bit symmetric: qmax = 7
    deq2[val][off] = __halves2half2(
       __int2half_rn((val & 0xF) - 7), __int2half_rn((val >> 4) - 7)
    );
  }

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  float res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
	float scale_f = scales[g * width + w];

    half2 scale = __float2half2_rn(scale_f);

    std::memset(&res2, 0, sizeof(half2));
    tmp = as_unsigned(mat[i]);
    res2 = __hfma2(__hmul2(deq2[(tmp >>  0) & 0xff][off], scale), blockvec[k + 0], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp >>  8) & 0xff][off], scale), blockvec[k + 1], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp >> 16) & 0xff][off], scale), blockvec[k + 2], res2);
    res2 = __hfma2(__hmul2(deq2[(tmp >> 24) & 0xff][off], scale), blockvec[k + 3], res2);
	i += width;
    k += 4;

    res += __low2float(res2) + __high2float(res2);
  }

  atomicAdd(&mul[b * width + w], res);
}


void vecquant2matmul_rtn(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, int groupsize, int vec_height
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_rtn_cuda(vec, mat, mul, scales, groupsize, vec_height);
}

void vecquant3matmul_rtn(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, int groupsize, int vec_height
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_rtn_cuda(vec, mat, mul, scales, groupsize, vec_height);
}

void vecquant4matmul_rtn(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, int groupsize, int vec_height
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_rtn_cuda(vec, mat, mul, scales, groupsize, vec_height);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant2matmul_rtn", &vecquant2matmul_rtn, "Vector 2-bit Quantized Matrix Multiplication (RTN Kernel)");
  m.def("vecquant3matmul_rtn", &vecquant3matmul_rtn, "Vector 3-bit Quantized Matrix Multiplication (RTN Kernel)");
  m.def("vecquant4matmul_rtn", &vecquant4matmul_rtn, "Vector 4-bit Quantized Matrix Multiplication (RTN Kernel)");
}
