#ifndef FLASHINFER_QUANTIZATION_CUH_
#define FLASHINFER_QUANTIZATION_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "vec_dtypes.cuh"

namespace flashinfer{
    namespace quant{

        /*!
        * \brief Identifier used for 4bit quantization,
        *   for data size calculattion.
        */
        struct __precision__s4{};

        /*!
        * \brief Simliar to sizeof
        * \tparam T Data type to be sizeof
        */
        template<typename T>
        FLASHINFER_INLINE constexpr float size_of_type(){
            if constexpr (std::is_same<T, __precision__s4>::value){
                return 0.5f;
            }else{
                return sizeof(T);
            }
        }

        /*!
        * \brief Used to get the pointer by offset.
        * \tparam T A template indicates the data type
        * \param ptr Pointer to the data
        * \param offset Offset to the pointer
        */
        template<typename T>
        FLASHINFER_INLINE T* get_ptr(T* ptr, const size_t offset){
            if constexpr (std::is_same<T, __precision__s4>::value){
                return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr) + offset / 2);
            }else if constexpr (std::is_same<T, const __precision__s4>::value){
                // Patch for const qualifiers
                return reinterpret_cast<T*>(reinterpret_cast<const uint8_t*>(ptr) + offset / 2);
            }else{
                return ptr + offset;
            }
        }

        /*!
        * \brief Dequantize the input into vec_t<float>
        * \tparam src_float_t A template indicates the quantization data type
        * \tparam vec_size A template integer indicates the vector size
        * \param src Const input data
        * \param tgt Output data
        * \param scale Quantization parameter
        * \param zero_point Quantization parameter
        */
        template<typename src_float_t, size_t vec_size>
        FLASHINFER_INLINE void dequantize_impl(
            const vec_t<src_float_t, vec_size> &src,
            vec_t<float, vec_size> &tgt,
            float scale,
            float zero_point
        ){
            if constexpr (std::is_same<src_float_t, __precision__s4>::value){
                // 4bit asymmetric quantization
                static_assert(vec_size % 8 == 0, "32bits pack 8 u4 elements.");
                // 8 x s4 in int32_t register
                constexpr size_t PACK_NUM = 8;
                #pragma unroll
                for(int i = 0;i < vec_size / PACK_NUM;++i){
                    uint32_t packedValue = src.at(i);
                    #pragma unroll
                    for(int j = 0; j < PACK_NUM;++j){
                        float unpackedValue = static_cast<float>(packedValue & 0xf) * scale - zero_point;
                        tgt[i * PACK_NUM + j] = unpackedValue;
                        packedValue >>= 4;
                    }
                }
            }else{
                // Not implemented
            }
        }
    }   // namespace quant
}   // namespace flashinfer

#endif  // FLASHINFER_QUANTIZATION_CUH_