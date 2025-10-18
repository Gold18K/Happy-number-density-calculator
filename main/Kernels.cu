
// Inclusions
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "cuda_runtime.h"
#include "Cuda_Utilities.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <iostream>
#include "Kernels.cuh"
#include <math.h>
#include "Memory.h"
#include <random>
#include <string>
#include <thread>

// Device functions
__device__ uint64_t Kernels::xorshift64(uint64_t& _state) {
    _state ^= _state >> 12;
    _state ^= _state << 25;
    _state ^= _state >> 27;

    return _state * 0x2545F4914F6CDD1DULL;
}
__device__ uint64_t Kernels::sum_squares(uint64_t _n,
                                         const uint16_t* _squares) {
    uint64_t sum = 0;

    while (_n != 0) {
        sum += _squares[_n % 10];
        _n /= 10;
    }

    return sum;
}

// Global functions
__global__ void Kernels::happy(const uint64_t* _seed,
                               const uint64_t* _n_of_digits,
                               const uint64_t* _batch_number_of_tests,
                               uint64_t* _happy_counter) {
    const uint64_t global_id   = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint16_t squares[10] = { 0,1,4,9,16,25,36,49,64,81 };

    if (global_id >= *_batch_number_of_tests)
		return;

    uint64_t local_seed = (*_seed ^ (global_id << 32)) + 0x9E3779B97F4A7C15ull;

    double u1 = (xorshift64(local_seed) >> 11) * (1 / (double)9007199254740992ull);
    double u2 = (xorshift64(local_seed) >> 11) * (1 / (double)9007199254740992ull);

    if (u1 < 1e-15)
        u1 = 1e-15;

    uint64_t slow = llround(fma(26.85 * sqrt((double)*_n_of_digits), sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2), 28.5 * *_n_of_digits));
    uint64_t fast = slow;

    if (slow == 0)
        return;

    do {
        slow = sum_squares(slow, squares);
		fast = sum_squares(sum_squares(fast, squares), squares);
    } while (slow != fast);

    if (slow == 1)
        atomicAdd(reinterpret_cast<unsigned long long int*>(_happy_counter), 1ull);
    
}

// Wrappers
void Kernels::launch_happy(const bool& _wait) {
    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    Memory<uint64_t> seed = Memory<uint64_t>(1);
    Memory<uint64_t> number_of_tests = Memory<uint64_t>(1);
    Memory<uint64_t> n_of_digits = Memory<uint64_t>(1);
    Memory<uint64_t> happy_counter = Memory<uint64_t>(1);

    number_of_tests[0] = 10000000000ull;

    Memory<uint64_t>::copy(number_of_tests, Cuda_Utilities::Unit::DEVICE, number_of_tests, Cuda_Utilities::Unit::HOST);

    for (uint64_t i = 3508294876; i <= 3508294876; i += 5) { // Change [from, to, step] as desired
        seed[0] = rng();
        n_of_digits[0] = i;
        happy_counter[0] = 0;

        Memory<uint64_t>::copy(seed, Cuda_Utilities::Unit::DEVICE, seed, Cuda_Utilities::Unit::HOST);
        Memory<uint64_t>::copy(n_of_digits, Cuda_Utilities::Unit::DEVICE, n_of_digits, Cuda_Utilities::Unit::HOST);
        Memory<uint64_t>::copy(happy_counter, Cuda_Utilities::Unit::DEVICE, happy_counter, Cuda_Utilities::Unit::HOST);

        const auto start_time = std::chrono::high_resolution_clock::now();

        happy<<<number_of_tests[0] / 512 + 1, 512, 0, 0>>>(seed.get_device_address(), n_of_digits.get_device_address(),
                number_of_tests.get_device_address(), happy_counter.get_device_address());

        Cuda_Utilities::cuda_synchronize();

        Memory<uint64_t>::copy(happy_counter, Cuda_Utilities::Unit::HOST, happy_counter, Cuda_Utilities::Unit::DEVICE);

        const auto end_time = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        std::cout << "Happy number density for " << n_of_digits[0] << " digits: " << static_cast<double>(happy_counter[0]) / (number_of_tests[0]) << "\n";
        std::cout << "Tests executed:   " << number_of_tests[0] << "\n";
        std::cout << "Computation time: " << duration << "ms\n\n";
    }

}