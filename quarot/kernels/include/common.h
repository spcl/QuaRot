#pragma once
#include <iostream>
#include <stdexcept>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

//#include <cutlass/subbyte_reference.h>
#include <int4.h>

#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__
#define HOST __forceinline__ __host__

HOST void ensure(bool condition, const std::string& msg) {
    if (!condition) {
        std::cerr << "Assertion failed: " << msg << '\n';
        // Choose the appropriate action: throw an exception, abort, etc.
        // For example, throwing an exception:
        throw std::runtime_error(msg);
    }
}

template<typename T>
HOST_DEVICE T mymax(T a, T b)
{
    return a > b ? a : b;
}

template<typename T>
HOST_DEVICE T mymin(T a, T b)
{
    return a < b ? a : b;
}

template<typename T>
HOST_DEVICE T cdiv(T a, T b) { return (a + b - 1) / b; }

template<typename T>
HOST_DEVICE T clamp(T x, T a, T b) { return mymax(a, mymin(b, x)); }

template<typename T>
HOST_DEVICE T myabs(T x) { return x < (T) 0 ? -x : x; }

template<typename T>
DEVICE T sqr(T x)
{   
    return x * x;
}

constexpr int qmin = -8;
constexpr int qmax = 7;




