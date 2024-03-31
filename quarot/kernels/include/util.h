#pragma once
/* TODO: This file can be safely discarded. Kept in case needed in the future.

// #include <bits/stdint-intn.h>
// #include <bits/stdint-uintn.h>
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/subbyte_reference.h>



#define _BITS_STDINT_UINTN_H	1
#define _BITS_STDINT_INTN_H	1
#include <bits/types.h>
typedef __int8_t int8_t;
typedef __int16_t int16_t;
typedef __int32_t int32_t;
typedef __int64_t int64_t;
typedef __uint8_t uint8_t;
typedef __uint16_t uint16_t;
typedef __uint32_t uint32_t;
typedef __uint64_t uint64_t;



template <typename T>
struct TorchDtypeDispatcher;

template <>
struct TorchDtypeDispatcher<uint8_t> {
  constexpr static const auto value = torch::kUInt8;
};

template <>
struct TorchDtypeDispatcher<int8_t> {
  constexpr static const auto value = torch::kInt8;
};

template <>
struct TorchDtypeDispatcher<int32_t> {
  constexpr static const auto value = torch::kInt32;
};

template <>
struct TorchDtypeDispatcher<cutlass::half_t> {
  constexpr static const auto value = torch::kFloat16;
};

template <typename T>
struct DtypeTorchDispatcher;

template <>
struct DtypeTorchDispatcher<torch::Half> {
  using value = __half;
};

template <>
struct DtypeTorchDispatcher<torch::BFloat16> {
  using value = __nv_bfloat16;
};

template <typename T>
__device__ inline int type2int_rn(T a) {
  return static_cast<int>(a);
}

template <>
__device__ inline int type2int_rn<__half>(__half input) {
  return __half2int_rn(input);
}

// template <>
// __device__ inline int type2int_rn<__nv_bfloat16>(__nv_bfloat16 input) {
//   return __bfloat162int_rn(input);
// }

template <typename T>
__device__ inline float type2float(T a) {
  return static_cast<float>(a);
}

template <>
__device__ inline float type2float<__half>(__half input) {
  return __half2float(input);
}

template <>
__device__ inline float type2float<__nv_bfloat16>(__nv_bfloat16 input) {
  return __bfloat162float(input);
}

template <typename T>
__device__ inline T float2type(float a) {
  return static_cast<float>(a);
}

template <>
__device__ inline __half float2type<__half>(float input) {
  return __float2half(input);
}

template <>
__device__ inline __nv_bfloat16 float2type<__nv_bfloat16>(float input) {
  return __float2bfloat16_rn(input);
}

template <typename T>
struct DtypeDtype2Dispatcher;

template <>
struct DtypeDtype2Dispatcher<__half> {
  using value = __half2;
};

template <>
struct DtypeDtype2Dispatcher<__nv_bfloat16> {
  using value = __nv_bfloat162;
};

__device__ inline __half2 type2type2(__half input, __half input2) {
  return __halves2half2(input, input2);
}

// __device__ inline __nv_bfloat162 type2type2(__nv_bfloat16 input,
//                                             __nv_bfloat16 input2) {
//   return __halves2bfloat162(input, input2);
// }

// template <typename T>
// T div(T a, T b) {
//   return a / b;
// }
//
// template <>
//__half div(__half a, __half b) {
//   return __hdiv(a, b);
// }
//
// template <>
//__nv_bfloat16 div(__nv_bfloat16 a, __nv_bfloat16 b) {
//   return __hdiv(a, b);
// }

*/