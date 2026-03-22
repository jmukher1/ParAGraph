#pragma once
#ifndef TIMING_GLOBALS_CUH
#define TIMING_GLOBALS_CUH
#include <cuda_runtime.h>

extern __device__ unsigned long long g_time_GetOneTwoHop_total;
extern __device__ unsigned long long g_time_GetOneTwoHop_count;
extern __device__ unsigned long long g_time_GetOneTwoHop_sumSquares;

extern __device__ unsigned long long g_time_SameYear_total;
extern __device__ unsigned long long g_time_SameYear_count;
extern __device__ unsigned long long g_time_SameYear_sumSquares;

extern __device__ unsigned long long g_time_MakeCitations_total;
extern __device__ unsigned long long g_time_MakeCitations_count;
extern __device__ unsigned long long g_time_MakeCitations_sumSquares;

extern __device__ unsigned long long g_time_UniformRandom_total;
extern __device__ unsigned long long g_time_UniformRandom_count;
extern __device__ unsigned long long g_time_UniformRandom_sumSquares;

#endif