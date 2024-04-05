#pragma once
#include <cuda_fp16.h>

#include <cstdint>

template <int head_dim>
void FlashInferBatchDecodeKernel_i4(nv_half* o, nv_half* q, void* kv_data,
                                    nv_half2* kv_param, int32_t* kv_indptr,
                                    int32_t* kv_indices,
                                    int32_t* last_page_offset, int num_layers,
                                    int layer_idx, int num_heads, int page_size,
                                    int batch_size);

template <int head_dim>
void FlashInferInitKvKernel_i4(void* kv_data, nv_half2* kv_param,
                               int32_t* kv_indptr, int32_t* kv_indices,
                               int32_t* last_page_offset, void* key,
                               void* value, nv_half2* key_param,
                               nv_half2* value_param, int32_t* seqlen_indptr,
                               int num_layers, int layer_idx, int num_heads,
                               int page_size, int batch_size);

template <int head_dim>
void FlashInferAppendKvKernel_i4(void* kv_data, nv_half2* kv_param,
                                 int32_t* kv_indptr, int32_t* kv_indices,
                                 int32_t* last_page_offset, void* key,
                                 void* value, nv_half2* key_param,
                                 nv_half2* value_param, int num_layers,
                                 int layer_idx, int num_heads, int page_size,
                                 int batch_size);


template <int head_dim>
void FlashInferBatchDecodeKernel_f16(nv_half* o, nv_half* q, void* kv_data,
                                    nv_half2* kv_param, int32_t* kv_indptr,
                                    int32_t* kv_indices,
                                    int32_t* last_page_offset, int num_layers,
                                    int layer_idx, int num_heads, int page_size,
                                    int batch_size);

template <int head_dim>
void FlashInferInitKvKernel_f16(void* kv_data, nv_half2* kv_param,
                               int32_t* kv_indptr, int32_t* kv_indices,
                               int32_t* last_page_offset, void* key,
                               void* value, nv_half2* key_param,
                               nv_half2* value_param, int32_t* seqlen_indptr,
                               int num_layers, int layer_idx, int num_heads,
                               int page_size, int batch_size);

template <int head_dim>
void FlashInferAppendKvKernel_f16(void* kv_data, nv_half2* kv_param,
                                 int32_t* kv_indptr, int32_t* kv_indices,
                                 int32_t* last_page_offset, void* key,
                                 void* value, nv_half2* key_param,
                                 nv_half2* value_param, int num_layers,
                                 int layer_idx, int num_heads, int page_size,
                                 int batch_size);
