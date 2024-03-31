#ifndef FLASHINFER_PAGE_CUH_
#define FLASHINFER_PAGE_CUH_

#include "layout.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"
#include "quantization.cuh"

namespace flashinfer {

/*!
 * \brief Paged key-value cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \note layout: [max_num_pages, num_layers, 2, num_heads, page_size, head_dim]
 * \note This layout is kind of HND, which is memory-friendly for the self-attn kernel's tile block.
 */
template <typename DType, typename IdType>
struct paged_kv_t {
  size_t num_layers;
  size_t layer_idx;
  size_t num_heads;
  size_t page_size;
  size_t head_dim;
  size_t batch_size;
  // [max_num_pages * num_layers * 2 * num_heads * page_size * head_dim]
  // The flattened key-value cache
  DType* data;
  // [max_num_pages * num_layers * 2 * num_heads * page_size * 1]
  // The flattened key-value quantization parameter cache
  half2* param;
  // [batch_size + 1] The page indptr array, with the first element 0
  IdType* indptr;
  // [nnz_pages] The page indices array
  IdType* indices;
  // [batch_size] The offset of the last page for each request in the batch
  IdType* last_page_offset;
  /*!
   * \brief Construct a paged key-value cache
   * \param num_layers The number of layers
   * \param layer_idx The index of the layer
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param data The flattened key-value cache
   * \param indptr The page indptr array
   * \param indices The page indices array
   * \param last_page_offset The offset of the last page for each request in the batch
   */
  __host__ __device__ __forceinline__ paged_kv_t(
    size_t num_layers,
    size_t layer_idx,
    size_t num_heads,
    size_t page_size,
    size_t head_dim,
    size_t batch_size,
    DType* data,
    half2* param,
    IdType* indptr,
    IdType* indices,
    IdType* last_page_offset
  ):  num_layers(num_layers),
      layer_idx(layer_idx),
      num_heads(num_heads),
      page_size(page_size),
      head_dim(head_dim),
      batch_size(batch_size),
      data(data),
      param(param),
      indptr(indptr),
      indices(indices),
      last_page_offset(last_page_offset) {}

  // \note layout: [max_num_pages, num_layers, 2, num_heads, page_size, head_dim]
  __host__ __device__ __forceinline__ size_t get_k_elem_offset(size_t page_idx, size_t head_idx,
                                                               size_t entry_idx, size_t feat_idx) {
    return (((page_idx * num_layers + layer_idx) * 2 * num_heads + head_idx) * page_size +
            entry_idx) *
               head_dim +
           feat_idx;
  }

  // \note layout: [max_num_pages, num_layers, 2, num_heads, page_size, head_dim]
  __host__ __device__ __forceinline__ size_t get_v_elem_offset(size_t page_idx, size_t head_idx,
                                                               size_t entry_idx, size_t feat_idx) {
    return ((((page_idx * num_layers + layer_idx) * 2 + 1) * num_heads + head_idx) * page_size +
            entry_idx) *
               head_dim +
           feat_idx;
  }

  // \note layout: [max_num_pages, num_layers, 2, num_heads, page_size, 1]
  __host__ __device__ __forceinline__ size_t get_param_k_elem_offset(size_t page_idx, size_t head_idx,
                                                               size_t entry_idx) {
    return ((page_idx * num_layers + layer_idx) * 2 * num_heads + head_idx) * page_size + entry_idx;
  }

  // \note layout: [max_num_pages, num_layers, 2, num_heads, page_size, 1]
  __host__ __device__ __forceinline__ size_t get_param_v_elem_offset(size_t page_idx, size_t head_idx,
                                                               size_t entry_idx) {
    return (((page_idx * num_layers + layer_idx) * 2 + 1) * num_heads + head_idx) * page_size + entry_idx;
  }

  __host__ __device__ __forceinline__ size_t get_valid_page_size(size_t batch_idx, size_t page_iter) {
    if (page_iter == indptr[batch_idx + 1] - 1) {
      return last_page_offset[batch_idx];
    } else {
      return page_size;
    }
  }
};

/*!
 * \brief: Append single token to the exisiting kv cache.
 * \note: Layout of key: [batch_size, num_heads, head_dim]
 * \note: this layout is natural output of previous dense layer, which don't need transpose.
*/
template <size_t head_dim, size_t vec_size, size_t bdx, size_t bdy, typename DType, typename IdType>
__global__ void AppendPagedKVCacheDecodeKernel(
  paged_kv_t<DType, IdType> paged_kv,
  DType* __restrict__ key,
  DType* __restrict__ value,
  half2* __restrict__ key_param,
  half2* __restrict__ value_param
) {
  size_t tx = threadIdx.x, ty = threadIdx.y;
  size_t num_heads = paged_kv.num_heads;
  size_t batch_idx = blockIdx.x / ((num_heads + bdy - 1) / bdy);
  size_t head_idx = (blockIdx.x % ((num_heads + bdy - 1) / bdy)) * bdy + ty;
  if (head_idx < num_heads) {
    // Pre-allocated enough space for the last page
    // seq_len included the added one
    size_t seq_len =
        (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
        paged_kv.last_page_offset[batch_idx];

    size_t page_idx =
        paged_kv.indices[paged_kv.indptr[batch_idx] + (seq_len - 1) / paged_kv.page_size];
    size_t entry_idx = (seq_len - 1) % paged_kv.page_size;

    vec_t<DType, vec_size>::memcpy(
        quant::get_ptr(paged_kv.data, paged_kv.get_k_elem_offset(page_idx, head_idx, entry_idx, tx * vec_size)),
        quant::get_ptr(key, (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size));

    vec_t<DType, vec_size>::memcpy(
        quant::get_ptr(paged_kv.data, paged_kv.get_v_elem_offset(page_idx, head_idx, entry_idx, tx * vec_size)),
        quant::get_ptr(value, (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size));
    
    // Copy the quantization parameters
    // One group only copies once
    if(tx == 0){
      quant::get_ptr(
        paged_kv.param,
        paged_kv.get_param_k_elem_offset(page_idx, head_idx, entry_idx)
      )[0] = key_param[batch_idx * num_heads + head_idx];

      quant::get_ptr(
        paged_kv.param,
        paged_kv.get_param_v_elem_offset(page_idx, head_idx, entry_idx)
      )[0] = value_param[batch_idx * num_heads + head_idx];
    }
  }
}

template <size_t head_dim, size_t vec_size, size_t bdx, size_t bdy, typename DType, typename IdType>
__global__ void AppendPagedKVCachePrefillKernel(
  paged_kv_t<DType, IdType> paged_kv,
  DType* __restrict__ key,
  DType* __restrict__ value,
  half2* __restrict__ key_param,
  half2* __restrict__ value_param,
  IdType* __restrict__ append_indptr
) {
  size_t tx = threadIdx.x, ty = threadIdx.y;
  size_t num_heads = paged_kv.num_heads;
  size_t batch_idx = blockIdx.x / ((num_heads + bdy - 1) / bdy);
  size_t head_idx = (blockIdx.x % ((num_heads + bdy - 1) / bdy)) * bdy + ty;
  if (head_idx < num_heads) {

    // Pre-filled seq_len
    size_t seq_len =
        (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
        paged_kv.last_page_offset[batch_idx];
    // Calculated to-be-filled seq_len
    size_t append_seq_len = append_indptr[batch_idx + 1] - append_indptr[batch_idx];
    size_t append_start = seq_len - append_seq_len;

#pragma unroll 2
    for (size_t j = 0; j < append_seq_len; ++j) {
      size_t page_seq_idx = j + append_start;
      size_t page_idx =
          paged_kv.indices[paged_kv.indptr[batch_idx] + page_seq_idx / paged_kv.page_size];
      size_t entry_idx = page_seq_idx % paged_kv.page_size;

      vec_t<DType, vec_size>::memcpy(
          quant::get_ptr(paged_kv.data, paged_kv.get_k_elem_offset(page_idx, head_idx, entry_idx, tx * vec_size)),
          quant::get_ptr(key, ((append_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size));

      vec_t<DType, vec_size>::memcpy(
          quant::get_ptr(paged_kv.data, paged_kv.get_v_elem_offset(page_idx, head_idx, entry_idx, tx * vec_size)),
          quant::get_ptr(value, ((append_indptr[batch_idx] + j) * num_heads + head_idx) * head_dim + tx * vec_size));
      
      // Copy the quantization parameters
      // One group only copies once
      if(tx == 0){
        quant::get_ptr(
          paged_kv.param,
          paged_kv.get_param_k_elem_offset(page_idx, head_idx, entry_idx)
        )[0] = key_param[(append_indptr[batch_idx] + j) * num_heads + head_idx];

        quant::get_ptr(
          paged_kv.param,
          paged_kv.get_param_v_elem_offset(page_idx, head_idx, entry_idx)
        )[0] = value_param[(append_indptr[batch_idx] + j) * num_heads + head_idx];
      }
    }
  }
}

template <typename DType, typename IdType>
cudaError_t AppendPagedKVCacheDecode(
  paged_kv_t<DType, IdType> paged_kv,
  DType* key,
  DType* value,
  half2* key_param,
  half2* value_param,
  cudaStream_t stream = nullptr,
  size_t dev_id = 0
) {
  FLASHINFER_CUDA_CALL(cudaSetDevice(dev_id));
  size_t head_dim = paged_kv.head_dim;
  size_t batch_size = paged_kv.batch_size;
  size_t num_heads = paged_kv.num_heads;
  SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr size_t vec_size = std::max(static_cast<size_t>(16 / quant::size_of_type<DType>()), HEAD_DIM / 32);
    constexpr size_t bdx = HEAD_DIM / vec_size;
    constexpr size_t bdy = 128 / bdx;
    assert(num_heads % bdy == 0);
    dim3 nblks(batch_size * num_heads / bdy);
    dim3 nthrs(bdx, bdy);
    auto kernel = AppendPagedKVCacheDecodeKernel<HEAD_DIM, vec_size, bdx, bdy, DType, IdType>;
    void* args[] = {
      (void*)&paged_kv,
      (void*)&key,
      (void*)&value,
      (void*)&key_param,
      (void*)&value_param
    };
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

template <typename DType, typename IdType>
cudaError_t AppendPagedKVCachePrefill(
  paged_kv_t<DType, IdType> paged_kv,
  DType* key,
  DType* value,
  half2* key_param,
  half2* value_param,
  IdType* append_indptr,
  cudaStream_t stream = nullptr,
  size_t dev_id = 0
) {
  FLASHINFER_CUDA_CALL(cudaSetDevice(dev_id));
  size_t head_dim = paged_kv.head_dim;
  size_t batch_size = paged_kv.batch_size;
  size_t num_heads = paged_kv.num_heads;
  SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr size_t vec_size = std::max(static_cast<size_t>(16 / quant::size_of_type<DType>()), HEAD_DIM / 32);
    constexpr size_t bdx = HEAD_DIM / vec_size;
    constexpr size_t bdy = 128 / bdx;
    assert(num_heads % bdy == 0);
    dim3 nblks(batch_size * num_heads / bdy);
    dim3 nthrs(bdx, bdy);
    auto kernel = AppendPagedKVCachePrefillKernel<HEAD_DIM, vec_size, bdx, bdy, DType, IdType>;
    void* args[] = {
      (void*)&paged_kv,
      (void*)&key,
      (void*)&value,
      (void*)&key_param,
      (void*)&value_param,
      (void*)&append_indptr
    };
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLAHSINFER_PAGE_CUH_
