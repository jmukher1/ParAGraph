#pragma once

#ifndef DEVICE_STRING_CUH
#define DEVICE_STRING_CUH

#include <cuda_runtime.h>

#define DEVICE_STRING_MAX_LEN 256

// -----------------------------------------------------------------------
// device_string
//
// All methods are defined INLINE, directly in the class body. This file
// was previously declaration-only for everything except the two
// constructors, with a commented-out `#include "device_string.cu"` at the
// bottom suggesting a separate implementation file that no longer exists
// (or never did). Since this header is included by multiple .cu
// translation units (graph.cu, abm.cu, kernel.cu, main.cu), any
// out-of-class definitions here MUST be inline (or the equivalent
// in-class-body implicit inline used below) to avoid "multiple definition"
// errors at link time -- this is why the fix keeps every body inside the
// class rather than reintroducing a separate .cu file.
//
// NOTE ON SEMANTICS: since no original implementation was available to
// recover, the bodies below implement standard, conservative C-string-like
// semantics for each named operation (see per-method comments). If your
// original device_string had different intended behavior for any edge
// case (e.g. scientific notation in to_float, case-insensitive equals),
// verify against your actual usage sites and adjust -- these are
// reasonable reconstructions, not a recovered original.
// -----------------------------------------------------------------------
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

        __host__ __device__ const char* c_str() const {
            return data;
        }

        __host__ __device__ void set(const char* str) {
            length = 0;
            for (int i = 0; i < DEVICE_STRING_MAX_LEN - 1 && str[i] != '\0'; ++i) {
                data[i] = str[i];
                ++length;
            }
            data[length] = '\0';
        }

        __host__ __device__ int size() const {
            return length;
        }

        __host__ __device__ bool empty() const {
            return length == 0;
        }

        __host__ __device__ char operator[](int index) const {
            // No bounds-check, matching the minimal/no-throw style of the
            // rest of this class (device code generally avoids exceptions).
            return data[index];
        }

        // Find the first occurrence of a character; return npos if not found
        __host__ __device__ int find(char ch) const {
            for (int i = 0; i < length; ++i) {
                if (data[i] == ch) return i;
            }
            return npos;
        }

        __host__ __device__ bool operator==(const device_string& other) const {
            if (length != other.length) return false;
            for (int i = 0; i < length; ++i) {
                if (data[i] != other.data[i]) return false;
            }
            return true;
        }

        // Append another device_string, truncating silently if it would
        // overflow DEVICE_STRING_MAX_LEN (consistent with the constructor's
        // silent-truncation behavior above).
        __host__ __device__ void append(const device_string& other) {
            for (int i = 0; i < other.length && length < DEVICE_STRING_MAX_LEN - 1; ++i) {
                data[length++] = other.data[i];
            }
            data[length] = '\0';
        }

        // Append from const char*
        __host__ __device__ void append(const char* str) {
            for (int i = 0; str[i] != '\0' && length < DEVICE_STRING_MAX_LEN - 1; ++i) {
                data[length++] = str[i];
            }
            data[length] = '\0';
        }

        // Append from const char
        __host__ __device__ void append_char(char c) {
            if (length < DEVICE_STRING_MAX_LEN - 1) {
                data[length++] = c;
                data[length] = '\0';
            }
        }

        // Standard itoa-style conversion, including negative numbers and 0.
        __host__ __device__ static device_string from_int(int val) {
            device_string result;
            if (val == 0) {
                result.data[0] = '0';
                result.data[1] = '\0';
                result.length = 1;
                return result;
            }

            bool negative = val < 0;
            // Note: this does not special-case INT_MIN (negating it
            // overflows); if you need to serialize INT_MIN specifically,
            // handle that case explicitly before calling from_int.
            unsigned int uval = negative ? (unsigned int)(-val) : (unsigned int)val;

            char buf[12];
            int buf_len = 0;
            while (uval > 0) {
                buf[buf_len++] = '0' + (uval % 10);
                uval /= 10;
            }
            if (negative) {
                result.data[result.length++] = '-';
            }
            // buf holds digits least-significant-first; reverse into data
            for (int i = buf_len - 1; i >= 0; --i) {
                result.data[result.length++] = buf[i];
            }
            result.data[result.length] = '\0';
            return result;
        }

        // Standard atoi-style parse: optional leading '-' or '+', digits
        // only. Stops at the first non-digit rather than erroring.
        __host__ __device__ int to_int() const {
            int i = 0;
            bool negative = false;
            if (i < length && (data[i] == '-' || data[i] == '+')) {
                negative = (data[i] == '-');
                ++i;
            }
            int result = 0;
            for (; i < length && data[i] >= '0' && data[i] <= '9'; ++i) {
                result = result * 10 + (data[i] - '0');
            }
            return negative ? -result : result;
        }

        // Standard decimal float parse: optional sign, integer part,
        // optional '.' + fractional part. No scientific-notation support.
        __host__ __device__ float to_float() const {
            int i = 0;
            bool negative = false;
            if (i < length && (data[i] == '-' || data[i] == '+')) {
                negative = (data[i] == '-');
                ++i;
            }
            float result = 0.0f;
            for (; i < length && data[i] >= '0' && data[i] <= '9'; ++i) {
                result = result * 10.0f + (data[i] - '0');
            }
            if (i < length && data[i] == '.') {
                ++i;
                float frac = 0.1f;
                for (; i < length && data[i] >= '0' && data[i] <= '9'; ++i) {
                    result += (data[i] - '0') * frac;
                    frac *= 0.1f;
                }
            }
            return negative ? -result : result;
        }

        // Convenience wrapper: parse another device_string's contents as a
        // float using this instance's to_float() logic.
        __host__ __device__ float device_string_to_float(device_string str) const {
            return str.to_float();
        }

        // Convenience wrapper: parse a raw C-string as a float by
        // constructing a temporary device_string and reusing to_float().
        __host__ __device__ float device_string_to_float(const char* str) const {
            device_string tmp(str);
            return tmp.to_float();
        }

        // -- Equals comparison with another device_string
        __host__ __device__ bool equals(const device_string& other) const {
            return (*this) == other;
        }

        // -- Equals comparison with C-string
        __host__ __device__ bool equals(const char* other) const {
            int i = 0;
            for (; i < length; ++i) {
                if (other[i] == '\0' || data[i] != other[i]) return false;
            }
            // Matched every character in `data`; make sure `other` doesn't
            // have extra trailing characters beyond length.
            return other[i] == '\0';
        }
};

#endif // DEVICE_STRING_CUH
