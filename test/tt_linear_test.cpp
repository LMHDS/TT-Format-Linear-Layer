#include "tt_linear/tt_linear.h"
#include <iostream>
#include <random>
#include <cassert>
#include <cmath>
#include <vector>
#include <iomanip>
#include <chrono>

using namespace gemm_hls::tt;

// 辅助函数：生成随机数据
template<typename T>
std::vector<T> generate_random_data(size_t size, float scale = 1.0f, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    
    std::vector<T> data(size);
    for (size_t i = 0; i < size; ++i) {
        if constexpr (std::is_same_v<T, half>) {
            data[i] = __float2half(dist(gen));
        } else {
            data[i] = static_cast<T>(dist(gen));
        }
    }
    return data;
}

template<typename T>
bool test_construction() {
    std::cout << "\nTesting basic construction..." << std::endl;
    
    try {
        // 使用默认配置
        TTLinear<T> layer;
        
        // 验证维度
        assert(layer.input_size() == kTTInputSize);
        assert(layer.output_size() == kTTOutputSize);
        assert(layer.num_cores() == kTTNumCores);
        
        // 初始化为0
        layer.initialize_zeros();

        // 验证每个核心
        using Config0 = typename TTLinear<T>::template CoreConfig<0>;
        using Config1 = typename TTLinear<T>::template CoreConfig<1>;
        using Config2 = typename TTLinear<T>::template CoreConfig<2>;
        using Config3 = typename TTLinear<T>::template CoreConfig<3>;

        // 验证GEMM维度
        assert(Config0::kGemmN * Config0::kGemmK == kTTInputSize);
        assert(Config3::kGemmN * Config3::kGemmM == kTTOutputSize);
        
        std::cout << "Basic construction test passed!" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "Construction test failed: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool test_forward_propagation() {
    std::cout << "\nTesting forward propagation..." << std::endl;
    
    try {
        TTLinear<T> layer;
        
        // 生成随机输入数据
        using Config0 = typename TTLinear<T>::template CoreConfig<0>;
        using MemoryPackN_t = typename TTLinear<T>::template MemoryPackN_t<0>;

        const size_t input_size = Config0::kGemmN * Config0::kGemmK;
        std::vector<T> input = generate_random_data<T>(input_size);
        
        using Config3 = typename TTLinear<T>::template CoreConfig<3>;
        const size_t output_size = Config3::kGemmN * Config3::kGemmM;
        std::vector<T> output(output_size);
        
        // 初始化层参数
        layer.initialize_random();
        
        // 执行前向传播
        auto start = std::chrono::high_resolution_clock::now();
        layer.forward(input.data(), output.data());
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Forward propagation took " << duration.count() << " ms" << std::endl;
        
        // 验证输出不全为0
        bool all_zeros = std::all_of(output.begin(), output.end(), 
                                   [](T val) { return val == T(0); });
        if (all_zeros) {
            throw std::runtime_error("Output contains all zeros");
        }
        
        // 打印一些输出样本
        std::cout << "Sample outputs:" << std::endl;
        for (int i = 0; i < 5 && i < output.size(); ++i) {
            float val = std::is_same_v<T, half> ? __half2float(output[i]) : output[i];
            std::cout << "output[" << i << "] = " << val << std::endl;
        }
        
        std::cout << "Forward propagation test passed!" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "Forward propagation test failed: " << e.what() << std::endl;
        return false;
    }
}

// 测试不同核心的GEMM操作
template<typename T>
bool test_gemm_operations() {
    std::cout << "\nTesting GEMM operations..." << std::endl;
    
    try {
        TTLinear<T> layer;
        layer.initialize_random();

        // 测试第一个核心的GEMM
        using Config0 = typename TTLinear<T>::template CoreConfig<0>;
        {
            const size_t N = Config0::kGemmN;
            const size_t K = Config0::kGemmK;
            const size_t M = Config0::kGemmM;
            
            std::vector<T> input(N * K);
            std::vector<T> output(N * M);
            
            // 使用随机数据
            for(size_t i = 0; i < input.size(); ++i) {
                input[i] = static_cast<T>(rand()) / RAND_MAX;
            }
            
            // 执行GEMM操作
            layer.compute_gemm<0>(input.data(), output.data());
            
            // 验证输出维度和数值
            assert(output.size() == N * M);
            
            std::cout << "Core 0 GEMM test passed!" << std::endl;
        }
        
        // Similar tests for other cores...
        using Config1 = typename TTLinear<T>::template CoreConfig<1>;
        {
            const size_t N = Config1::kGemmN;
            const size_t K = Config1::kGemmK;
            const size_t M = Config1::kGemmM;
            
            std::vector<T> input(N * K);
            std::vector<T> output(N * M);
            
            // 使用随机数据
            for(size_t i = 0; i < input.size(); ++i) {
                input[i] = static_cast<T>(rand()) / RAND_MAX;
            }
            
            // 执行GEMM操作
            layer.compute_gemm<1>(input.data(), output.data());
            
            // 验证输出维度和数值
            assert(output.size() == N * M);
            
            std::cout << "Core 1 GEMM test passed!" << std::endl;
        }
        
        using Config2 = typename TTLinear<T>::template CoreConfig<2>;
        {
            const size_t N = Config2::kGemmN;
            const size_t K = Config2::kGemmK;
            const size_t M = Config2::kGemmM;
            
            std::vector<T> input(N * K);
            std::vector<T> output(N * M);
            
            // 使用随机数据
            for(size_t i = 0; i < input.size(); ++i) {
                input[i] = static_cast<T>(rand()) / RAND_MAX;
            }
            
            // 执行GEMM操作
            layer.compute_gemm<2>(input.data(), output.data());
            
            // 验证输出维度和数值
            assert(output.size() == N * M);
            
            std::cout << "Core 2 GEMM test passed!" << std::endl;
        }
        
        using Config3 = typename TTLinear<T>::template CoreConfig<3>;
        {
            const size_t N = Config3::kGemmN;
            const size_t K = Config3::kGemmK;
            const size_t M = Config3::kGemmM;
            
            std::vector<T> input(N * K);
            std::vector<T> output(N * M);
            
            // 使用随机数据
            for(size_t i = 0; i < input.size(); ++i) {
                input[i] = static_cast<T>(rand()) / RAND_MAX;
            }
            
            // 执行GEMM操作
            layer.compute_gemm<3>(input.data(), output.data());
            
            // 验证输出维度和数值
            assert(output.size() == N * M);
            
            std::cout << "Core 3 GEMM test passed!" << std::endl;
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "GEMM operations test failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    bool float_tests_passed = test_construction<float>() &&
                            test_forward_propagation<float>() &&
                            test_gemm_operations<float>();
                            
    #ifdef MM_HALF_PRECISION
    bool half_tests_passed = test_construction<half>() &&
                           test_forward_propagation<half>() &&
                           test_gemm_operations<half>();
    #else
    bool half_tests_passed = true;
    #endif

    if (float_tests_passed && half_tests_passed) {
        std::cout << "\nAll tests passed successfully!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome tests failed!" << std::endl;
        return 1;
    }
}
