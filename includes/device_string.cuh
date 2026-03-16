#pragma once

#ifndef DEVICE_STRING_CUH
#define DEVICE_STRING_CUH

#include <cuda_runtime.h>

#define DEVICE_STRING_MAX_LEN 256

 
class device_string {
    public:
        char data[DEVICE_STRING_MAX_LEN];
        int length;

        static const int npos = -1;

        __host__ __device__ device_string() : length(0) {
            data[0] = '\0';
        }

        __host__ __device__ device_string(const char* str) {
            length = 0;
            for (int i = 0; i < DEVICE_STRING_MAX_LEN - 1 && str[i] != '\0'; ++i) {
                data[i] = str[i];
                ++length;
            }
            data[length] = '\0';
        }

        __host__ __device__ const char* c_str() const;

        __host__ __device__ void set(const char* str);

        __host__ __device__ int size() const;

        __host__ __device__ bool empty() const;

        __host__ __device__ char operator[](int index) const;

        // Find the first occurrence of a character; return npos if not found
        __host__ __device__ int find(char ch) const;

        __host__ __device__ bool operator==(const device_string& other) const;

        // Append another device_string
        __host__ __device__ void append(const device_string& other);

        // Append from const char*
        __host__ __device__ void append(const char* str);

        // Append from const char
        __host__ __device__ void append_char(char c);

        __host__ __device__ static device_string from_int(int val);

        __host__ __device__ int to_int() const;

        __host__ __device__ float to_float() const;

        __host__ __device__ float device_string_to_float(device_string str) const;
        
        __host__ __device__ float device_string_to_float(const char* str) const;

        // -- Equals comparison with another device_string
        __host__ __device__ bool equals(const device_string& other) const;

        // -- Equals comparison with C-string
        __host__ __device__ bool equals(const char* other) const;
};

//#include "device_string.cu"

#endif // DEVICE_STRING_CUH