#pragma once

// Inclusions
#include <atomic>
#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace Kernels {

    __device__ uint64_t xorshift64(uint64_t& _state);
    __device__ uint64_t sum_squares(uint64_t _n,
                                    const uint16_t* _squares);
    __global__ void     happy(const uint64_t* _seed,
                              const uint64_t* _n_of_digits,
                              const uint64_t* _batch_number_of_tests,
                              uint64_t* _happy_counter);
	void                launch_happy(const bool& _wait = true);

}
