#pragma once
#include <algorithm>
#include <cmath>

#include <flashinfer/decode.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/quantization.cuh>

template <int head_dim>
void FlashInferBatchDecodeKernel_i4(nv_half* o, nv_half* q, void* kv_data,
                                    nv_half2* kv_param, int32_t* kv_indptr,
                                    int32_t* kv_indicies,
                                    int32_t* last_page_offset, int num_layers,
                                    int layer_idx, int num_heads, int page_size,
                                    int batch_size) {
  using DTypeIn = flashinfer::quant::__precision__s4;
  using DTypeInQ = nv_half;
  using DTypeOut = nv_half;

  flashinfer::paged_kv_t<DTypeIn, int32_t> paged_kv(
      num_layers, layer_idx, num_heads, page_size, head_dim, batch_size,
      (DTypeIn*)kv_data, kv_param, kv_indptr, kv_indicies, last_page_offset);

  const float rope_scale = 1.f;
  const float rope_theta = 1e4;
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  const float rope_inv_scale = 1.f / rope_scale;
  const float rope_inv_theta = 1.f / rope_theta;

  constexpr bool norm_on_the_fly = false;
  constexpr auto rotary_mode = flashinfer::RotaryMode::kNone;
  constexpr size_t FoldFactor = 2;
  constexpr size_t vec_size = std::max(
      static_cast<size_t>(16 / flashinfer::quant::size_of_type<DTypeIn>() /
                          FoldFactor),
      static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 128 / bdx;
  dim3 nblks(paged_kv.batch_size, paged_kv.num_heads);
  dim3 nthrs(bdx, bdy);

  flashinfer::BatchDecodeWithPagedKVCacheKernel<
      rotary_mode, norm_on_the_fly, vec_size, bdx, bdy, FoldFactor, DTypeInQ,
      DTypeIn, DTypeOut, int32_t><<<nblks, nthrs>>>(
      q, paged_kv, o, sm_scale, rope_inv_scale, rope_inv_theta);
}

template <int head_dim>
void FlashInferInitKvKernel_i4(void* kv_data, nv_half2* kv_param,
                               int32_t* kv_indptr, int32_t* kv_indicies,
                               int32_t* last_page_offset, void* key,
                               void* value, nv_half2* key_param,
                               nv_half2* value_param, int32_t* seqlen_indptr,
                               int num_layers, int layer_idx, int num_heads,
                               int page_size, int batch_size) {
  using T = flashinfer::quant::__precision__s4;
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_heads, page_size, head_dim, batch_size,
      (T*)kv_data, kv_param, kv_indptr, kv_indicies, last_page_offset);

  constexpr size_t vec_size =
      std::max(static_cast<size_t>(16 / flashinfer::quant::size_of_type<T>()),
               static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 128 / bdx;
  dim3 nblks(paged_kv.batch_size * ((paged_kv.num_heads + bdy - 1) / bdy));
  dim3 nthrs(bdx, bdy);
  flashinfer::AppendPagedKVCachePrefillKernel<head_dim, vec_size, bdx, bdy, T,
                                              int32_t><<<nblks, nthrs>>>(
      paged_kv, (T*)key, (T*)value, key_param, value_param, seqlen_indptr);
}

template <int head_dim>
void FlashInferAppendKvKernel_i4(void* kv_data, nv_half2* kv_param,
                                 int32_t* kv_indptr, int32_t* kv_indicies,
                                 int32_t* last_page_offset, void* key,
                                 void* value, nv_half2* key_param,
                                 nv_half2* value_param, int num_layers,
                                 int layer_idx, int num_heads, int page_size,
                                 int batch_size) {
  using T = flashinfer::quant::__precision__s4;
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_heads, page_size, head_dim, batch_size,
      (T*)kv_data, kv_param, kv_indptr, kv_indicies, last_page_offset);

  constexpr size_t vec_size =
      std::max(static_cast<size_t>(16 / flashinfer::quant::size_of_type<T>()),
               static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 128 / bdx;
  dim3 nblks(paged_kv.batch_size * ((paged_kv.num_heads + bdy - 1) / bdy));
  dim3 nthrs(bdx, bdy);
  flashinfer::AppendPagedKVCacheDecodeKernel<head_dim, vec_size, bdx, bdy, T,
                                             int32_t>
      <<<nblks, nthrs>>>(paged_kv, (T*)key, (T*)value, key_param, value_param);
}


template <int head_dim>
void FlashInferBatchDecodeKernel_f16(nv_half* o, nv_half* q, void* kv_data,
                                    nv_half2* kv_param, int32_t* kv_indptr,
                                    int32_t* kv_indicies,
                                    int32_t* last_page_offset, int num_layers,
                                    int layer_idx, int num_heads, int page_size,
                                    int batch_size) {
  using DTypeIn = nv_half;
  using DTypeInQ = nv_half;
  using DTypeOut = nv_half;

  flashinfer::paged_kv_t<DTypeIn, int32_t> paged_kv(
      num_layers, layer_idx, num_heads, page_size, head_dim, batch_size,
      (DTypeIn*)kv_data, kv_param, kv_indptr, kv_indicies, last_page_offset);

  const float rope_scale = 1.f;
  const float rope_theta = 1e4;
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  const float rope_inv_scale = 1.f / rope_scale;
  const float rope_inv_theta = 1.f / rope_theta;

  constexpr bool norm_on_the_fly = false;
  constexpr auto rotary_mode = flashinfer::RotaryMode::kNone;
  constexpr size_t FoldFactor = 1;
  constexpr size_t vec_size = std::max(
      static_cast<size_t>(16 / flashinfer::quant::size_of_type<DTypeIn>() /
                          FoldFactor),
      static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 128 / bdx;
  dim3 nblks(paged_kv.batch_size, paged_kv.num_heads);
  dim3 nthrs(bdx, bdy);

  flashinfer::BatchDecodeWithPagedKVCacheKernel<
      rotary_mode, norm_on_the_fly, vec_size, bdx, bdy, FoldFactor, DTypeInQ,
      DTypeIn, DTypeOut, int32_t><<<nblks, nthrs>>>(
      q, paged_kv, o, sm_scale, rope_inv_scale, rope_inv_theta);
}

template <int head_dim>
void FlashInferInitKvKernel_f16(void* kv_data, nv_half2* kv_param,
                               int32_t* kv_indptr, int32_t* kv_indicies,
                               int32_t* last_page_offset, void* key,
                               void* value, nv_half2* key_param,
                               nv_half2* value_param, int32_t* seqlen_indptr,
                               int num_layers, int layer_idx, int num_heads,
                               int page_size, int batch_size) {
  using T = nv_half;
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_heads, page_size, head_dim, batch_size,
      (T*)kv_data, kv_param, kv_indptr, kv_indicies, last_page_offset);

  constexpr size_t vec_size =
      std::max(static_cast<size_t>(16 / flashinfer::quant::size_of_type<T>()),
               static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 128 / bdx;
  dim3 nblks(paged_kv.batch_size * ((paged_kv.num_heads + bdy - 1) / bdy));
  dim3 nthrs(bdx, bdy);
  flashinfer::AppendPagedKVCachePrefillKernel<head_dim, vec_size, bdx, bdy, T,
                                              int32_t><<<nblks, nthrs>>>(
      paged_kv, (T*)key, (T*)value, key_param, value_param, seqlen_indptr);
}

template <int head_dim>
void FlashInferAppendKvKernel_f16(void* kv_data, nv_half2* kv_param,
                                 int32_t* kv_indptr, int32_t* kv_indicies,
                                 int32_t* last_page_offset, void* key,
                                 void* value, nv_half2* key_param,
                                 nv_half2* value_param, int num_layers,
                                 int layer_idx, int num_heads, int page_size,
                                 int batch_size) {
  using T = nv_half;
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_heads, page_size, head_dim, batch_size,
      (T*)kv_data, kv_param, kv_indptr, kv_indicies, last_page_offset);

  constexpr size_t vec_size =
      std::max(static_cast<size_t>(16 / flashinfer::quant::size_of_type<T>()),
               static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 128 / bdx;
  dim3 nblks(paged_kv.batch_size * ((paged_kv.num_heads + bdy - 1) / bdy));
  dim3 nthrs(bdx, bdy);
  flashinfer::AppendPagedKVCacheDecodeKernel<head_dim, vec_size, bdx, bdy, T,
                                             int32_t>
      <<<nblks, nthrs>>>(paged_kv, (T*)key, (T*)value, key_param, value_param);
}


template void FlashInferBatchDecodeKernel_i4<128>(
    nv_half* o, nv_half* q, void* kv_data, nv_half2* kv_param,
    int32_t* kv_indptr, int32_t* kv_indicies, int32_t* last_page_offset,
    int num_layers, int layer_idx, int num_heads, int page_size,
    int batch_size);

template void FlashInferInitKvKernel_i4<128>(
    void* kv_data, nv_half2* kv_param, int32_t* kv_indptr, int32_t* kv_indicies,
    int32_t* last_page_offset, void* key, void* value, nv_half2* key_param,
    nv_half2* value_param, int32_t* seqlen_indptr, int num_layers,
    int layer_idx, int num_heads, int page_size, int batch_size);

template void FlashInferAppendKvKernel_i4<128>(
    void* kv_data, nv_half2* kv_param, int32_t* kv_indptr, int32_t* kv_indicies,
    int32_t* last_page_offset, void* key, void* value, nv_half2* key_param,
    nv_half2* value_param, int num_layers, int layer_idx, int num_heads,
    int page_size, int batch_size);


template void FlashInferBatchDecodeKernel_f16<128>(
    nv_half* o, nv_half* q, void* kv_data, nv_half2* kv_param,
    int32_t* kv_indptr, int32_t* kv_indicies, int32_t* last_page_offset,
    int num_layers, int layer_idx, int num_heads, int page_size,
    int batch_size);

template void FlashInferInitKvKernel_f16<128>(
    void* kv_data, nv_half2* kv_param, int32_t* kv_indptr, int32_t* kv_indicies,
    int32_t* last_page_offset, void* key, void* value, nv_half2* key_param,
    nv_half2* value_param, int32_t* seqlen_indptr, int num_layers,
    int layer_idx, int num_heads, int page_size, int batch_size);

template void FlashInferAppendKvKernel_f16<128>(
    void* kv_data, nv_half2* kv_param, int32_t* kv_indptr, int32_t* kv_indicies,
    int32_t* last_page_offset, void* key, void* value, nv_half2* key_param,
    nv_half2* value_param, int num_layers, int layer_idx, int num_heads,
    int page_size, int batch_size);
