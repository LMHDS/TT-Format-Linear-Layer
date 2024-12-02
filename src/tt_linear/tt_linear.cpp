#include "tt_linear/tt_linear.h"
#include <numeric>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace gemm_hls {
namespace tt {

template<typename T>
void TTLinear<T>::Config::initialize_default() {
    #ifdef TT_USE_CUSTOM_MODES
        input_modes = std::vector<unsigned>(kTTInputModes.begin(), kTTInputModes.end());
        output_modes = std::vector<unsigned>(kTTOutputModes.begin(), kTTOutputModes.end());
        ranks = std::vector<unsigned>(kTTRanks.begin(), kTTRanks.end());
    #else
        input_modes.resize(num_cores);
        output_modes.resize(num_cores);
        ranks.resize(num_cores + 1);

        // 设置维度分解
        input_modes = {8, 8, 8, 2};   // 1024 = 8*8*8*2
        output_modes = {8, 8, 4, 2};  // 512 = 8*8*4*2
        
        // 设置TT秩 [1, 64, 64, 64, 1]
        ranks[0] = ranks[num_cores] = 1;
        std::fill(ranks.begin() + 1, ranks.end() - 1, kTTMaxRank);
    #endif
    validate();
}

template<typename T>
TTLinear<T>::TTLinear(const Config& config) : config_(config) {
    initialize_cores();
    initialize_buffers();
}

template<typename T>
void TTLinear<T>::initialize_cores() {
    // 初始化每个核心
    core0_ = std::make_unique<TTCore<T, 0>>(
        config_.ranks[0], config_.input_modes[0],
        config_.ranks[1], config_.output_modes[0]);
        
    core1_ = std::make_unique<TTCore<T, 1>>(
        config_.ranks[1], config_.input_modes[1],
        config_.ranks[2], config_.output_modes[1]);
        
    core2_ = std::make_unique<TTCore<T, 2>>(
        config_.ranks[2], config_.input_modes[2],
        config_.ranks[3], config_.output_modes[2]);
        
    core3_ = std::make_unique<TTCore<T, 3>>(
        config_.ranks[3], config_.input_modes[3],
        config_.ranks[4], config_.output_modes[3]);
    
    verify_chain_consistency();
}

template<typename T>
void TTLinear<T>::initialize_buffers() {
    // 初始化每个GEMM的缓冲区
    for(int i = 0; i < 4; ++i) {
        switch(i) {
            case 0: {
                using Config = tt_core0;
                const size_t size_A = (Config::kGemmN * Config::kGemmK + Config::kMemoryWidthN - 1) 
                                    / Config::kMemoryWidthN;
                const size_t size_B = (Config::kGemmK * Config::kGemmM + Config::kMemoryWidthM - 1) 
                                    / Config::kMemoryWidthM;
                const size_t size_C = (Config::kGemmN * Config::kGemmM + Config::kMemoryWidthM - 1) 
                                    / Config::kMemoryWidthM;
                gemm_buffers_[0].input_A.resize(size_A);
                gemm_buffers_[0].weight_B.resize(size_B);
                gemm_buffers_[0].output_C.resize(size_C);
                gemm_buffers_[0].reshape_buffer.resize(Config::kGemmN * Config::kGemmM);
                break;
            }
            case 1: {
                using Config = tt_core1;
                const size_t size_A = (Config::kGemmN * Config::kGemmK + Config::kMemoryWidthN - 1) 
                                    / Config::kMemoryWidthN;
                const size_t size_B = (Config::kGemmK * Config::kGemmM + Config::kMemoryWidthM - 1) 
                                    / Config::kMemoryWidthM;
                const size_t size_C = (Config::kGemmN * Config::kGemmM + Config::kMemoryWidthM - 1) 
                                    / Config::kMemoryWidthM;
                gemm_buffers_[1].input_A.resize(size_A);
                gemm_buffers_[1].weight_B.resize(size_B);
                gemm_buffers_[1].output_C.resize(size_C);
                gemm_buffers_[1].reshape_buffer.resize(Config::kGemmN * Config::kGemmM);
                break;
            }
            case 2: {
                using Config = tt_core2;
                const size_t size_A = (Config::kGemmN * Config::kGemmK + Config::kMemoryWidthN - 1) 
                                    / Config::kMemoryWidthN;
                const size_t size_B = (Config::kGemmK * Config::kGemmM + Config::kMemoryWidthM - 1) 
                                    / Config::kMemoryWidthM;
                const size_t size_C = (Config::kGemmN * Config::kGemmM + Config::kMemoryWidthM - 1) 
                                    / Config::kMemoryWidthM;
                gemm_buffers_[2].input_A.resize(size_A);
                gemm_buffers_[2].weight_B.resize(size_B);
                gemm_buffers_[2].output_C.resize(size_C);
                gemm_buffers_[2].reshape_buffer.resize(Config::kGemmN * Config::kGemmM);
                break;
            }
            case 3: {
                using Config = tt_core3;
                const size_t size_A = (Config::kGemmN * Config::kGemmK + Config::kMemoryWidthN - 1) 
                                    / Config::kMemoryWidthN;
                const size_t size_B = (Config::kGemmK * Config::kGemmM + Config::kMemoryWidthM - 1) 
                                    / Config::kMemoryWidthM;
                const size_t size_C = (Config::kGemmN * Config::kGemmM + Config::kMemoryWidthM - 1) 
                                    / Config::kMemoryWidthM;
                gemm_buffers_[3].input_A.resize(size_A);
                gemm_buffers_[3].weight_B.resize(size_B);
                gemm_buffers_[3].output_C.resize(size_C);
                gemm_buffers_[3].reshape_buffer.resize(Config::kGemmN * Config::kGemmM);
                break;
            }
        }
    }
}
template<typename T>
void TTLinear<T>::forward(const T* input, T* output) {
    // 1. First GEMM: [I2*I3*I4, I1] × [I1, r1*O1]
    {
        using Config = tt_core0;
        // 准备输入矩阵
        prepare_input_matrix<0>(input, gemm_buffers_[0].input_A);
        // 准备权重矩阵
        core0_->prepare_for_gemm(gemm_buffers_[0].weight_B);
        
        // 执行GEMM
        MatrixMultiplicationKernel<T>(
            gemm_buffers_[0].input_A.data(),
            gemm_buffers_[0].weight_B.data(),
            gemm_buffers_[0].output_C.data(),
            Config::kGemmN,
            Config::kGemmK,
            Config::kGemmM
        );

        // 重组输出为下一个GEMM的输入
        reshape_output_to_input<0>(
            gemm_buffers_[0].output_C,
            gemm_buffers_[1].input_A
        );
    }

    // 2. Second GEMM: [O1*I3*I4, r1*I2] × [r1*I2, r2*O2]
    {
        using Config = tt_core1;
        core1_->prepare_for_gemm(gemm_buffers_[1].weight_B);
        
        MatrixMultiplicationKernel<T>(
            gemm_buffers_[1].input_A.data(),
            gemm_buffers_[1].weight_B.data(),
            gemm_buffers_[1].output_C.data(),
            Config::kGemmN,
            Config::kGemmK,
            Config::kGemmM
        );

        reshape_output_to_input<1>(
            gemm_buffers_[1].output_C,
            gemm_buffers_[2].input_A
        );
    }

    // 3. Third GEMM: [O1*O2*I4, r2*I3] × [r2*I3, r3*O3]
    {
        using Config = tt_core2;
        core2_->prepare_for_gemm(gemm_buffers_[2].weight_B);
        
        MatrixMultiplicationKernel<T>(
            gemm_buffers_[2].input_A.data(),
            gemm_buffers_[2].weight_B.data(),
            gemm_buffers_[2].output_C.data(),
            Config::kGemmN,
            Config::kGemmK,
            Config::kGemmM
        );

        reshape_output_to_input<2>(
            gemm_buffers_[2].output_C,
            gemm_buffers_[3].input_A
        );
    }

    // 4. Fourth GEMM: [O1*O2*O3, r3*I4] × [r3*I4, O4]
    {
        using Config = tt_core3;
        core3_->prepare_for_gemm(gemm_buffers_[3].weight_B);
        
        MatrixMultiplicationKernel<T>(
            gemm_buffers_[3].input_A.data(),
            gemm_buffers_[3].weight_B.data(),
            gemm_buffers_[3].output_C.data(),
            Config::kGemmN,
            Config::kGemmK,
            Config::kGemmM
        );

        // 准备最终输出
        prepare_final_output<3>(gemm_buffers_[3].output_C, output);
    }
}
template<typename T>
template<int CoreIndex>
void TTLinear<T>::prepare_input_matrix(const T* input, 
                                     typename GemmBuffers::MemoryPackN_t& matrix) const {
    if constexpr (CoreIndex == 0) {
        using Config = tt_core0;
        const unsigned N = Config::kGemmN;  // I2*I3*I4
        const unsigned K = Config::kGemmK;  // I1
        
        #pragma HLS PIPELINE II=1
        for (unsigned n = 0; n < N; ++n) {
            for (unsigned k = 0; k < K; ++k) {
                const unsigned dst_idx = n * K + k;
                const unsigned pack_idx = dst_idx / Config::kMemoryWidthN;
                const unsigned pack_pos = dst_idx % Config::kMemoryWidthN;
                
                if (pack_pos == 0) {
                    matrix[pack_idx] = typename Config::MemoryPackN_t();
                }
                matrix[pack_idx][pack_pos] = input[k * N + n];
            }
        }
    }
}

template<typename T>
template<int CoreIndex>
void TTLinear<T>::reshape_output_to_input(
    const typename GemmBuffers::MemoryPackM_t& curr_output,
    typename GemmBuffers::MemoryPackN_t& next_input) const {
    
    using SrcConfig = typename std::conditional<CoreIndex == 0, tt_core0,
                     typename std::conditional<CoreIndex == 1, tt_core1,
                     typename std::conditional<CoreIndex == 2, tt_core2,
                                                           tt_core3>::type>::type>::type;
                                                           
    using DstConfig = typename std::conditional<CoreIndex == 0, tt_core1,
                     typename std::conditional<CoreIndex == 1, tt_core2,
                     typename std::conditional<CoreIndex == 2, tt_core3,
                                                           tt_core3>::type>::type>::type;

    // 1. Unpack current output to reshape buffer
    #pragma HLS PIPELINE II=1
    for (unsigned i = 0; i < SrcConfig::kGemmN; ++i) {
        for (unsigned j = 0; j < SrcConfig::kGemmM; ++j) {
            const unsigned src_idx = i * SrcConfig::kGemmM + j;
            const unsigned src_pack = src_idx / SrcConfig::kMemoryWidthM;
            const unsigned src_pos = src_idx % SrcConfig::kMemoryWidthM;
            gemm_buffers_[CoreIndex].reshape_buffer[src_idx] = 
                curr_output[src_pack][src_pos];
        }
    }

    // 2. Reshape and pack for next GEMM
    constexpr unsigned O1 = config_.output_modes[0];
    constexpr unsigned O2 = config_.output_modes[1];
    constexpr unsigned O3 = config_.output_modes[2];
    constexpr unsigned I2 = config_.input_modes[1];
    constexpr unsigned I3 = config_.input_modes[2];
    constexpr unsigned I4 = config_.input_modes[3];
    constexpr unsigned r1 = config_.ranks[1];
    constexpr unsigned r2 = config_.ranks[2];
    constexpr unsigned r3 = config_.ranks[3];

    #pragma HLS PIPELINE II=1
    if constexpr (CoreIndex == 0) {
        // [I2*I3*I4, r1*O1] -> [O1*I3*I4, r1*I2]
        for (unsigned o1 = 0; o1 < O1; ++o1) {
            for (unsigned i3 = 0; i3 < I3; ++i3) {
                for (unsigned i4 = 0; i4 < I4; ++i4) {
                    for (unsigned r = 0; r < r1; ++r) {
                        for (unsigned i2 = 0; i2 < I2; ++i2) {
                            const unsigned src_row = i2 * I3 * I4 + i3 * I4 + i4;
                            const unsigned src_col = r * O1 + o1;
                            const unsigned src_idx = src_row * (r1 * O1) + src_col;

                            const unsigned dst_row = o1 * I3 * I4 + i3 * I4 + i4;
                            const unsigned dst_col = r * I2 + i2;
                            const unsigned dst_idx = dst_row * (r1 * I2) + dst_col;
                            
                            const unsigned dst_pack = dst_idx / DstConfig::kMemoryWidthN;
                            const unsigned dst_pos = dst_idx % DstConfig::kMemoryWidthN;
                            
                            if (dst_pos == 0) {
                                next_input[dst_pack] = typename DstConfig::MemoryPackN_t();
                            }
                            next_input[dst_pack][dst_pos] = 
                                gemm_buffers_[CoreIndex].reshape_buffer[src_idx];
                        }
                    }
                }
            }
        }
    }
    else if constexpr (CoreIndex == 1) {
        // [O1*I3*I4, r2*O2] -> [O1*O2*I4, r2*I3]
        for (unsigned o1 = 0; o1 < O1; ++o1) {
            for (unsigned o2 = 0; o2 < O2; ++o2) {
                for (unsigned i4 = 0; i4 < I4; ++i4) {
                    for (unsigned r = 0; r < r2; ++r) {
                        for (unsigned i3 = 0; i3 < I3; ++i3) {
                            const unsigned src_row = o1 * I3 * I4 + i3 * I4 + i4;
                            const unsigned src_col = r * O2 + o2;
                            const unsigned src_idx = src_row * (r2 * O2) + src_col;

                            const unsigned dst_row = o1 * O2 * I4 + o2 * I4 + i4;
                            const unsigned dst_col = r * I3 + i3;
                            const unsigned dst_idx = dst_row * (r2 * I3) + dst_col;
                            
                            const unsigned dst_pack = dst_idx / DstConfig::kMemoryWidthN;
                            const unsigned dst_pos = dst_idx % DstConfig::kMemoryWidthN;
                            
                            if (dst_pos == 0) {
                                next_input[dst_pack] = typename DstConfig::MemoryPackN_t();
                            }
                            next_input[dst_pack][dst_pos] = 
                                gemm_buffers_[CoreIndex].reshape_buffer[src_idx];
                        }
                    }
                }
            }
        }
    }
    else if constexpr (CoreIndex == 2) {
        // [O1*O2*I4, r3*O3] -> [O1*O2*O3, r3*I4]
        for (unsigned o1 = 0; o1 < O1; ++o1) {
            for (unsigned o2 = 0; o2 < O2; ++o2) {
                for (unsigned o3 = 0; o3 < O3; ++o3) {
                    for (unsigned r = 0; r < r3; ++r) {
                        for (unsigned i4 = 0; i4 < I4; ++i4) {
                            const unsigned src_row = o1 * O2 * I4 + o2 * I4 + i4;
                            const unsigned src_col = r * O3 + o3;
                            const unsigned src_idx = src_row * (r3 * O3) + src_col;

                            const unsigned dst_row = o1 * O2 * O3 + o2 * O3 + o3;
                            const unsigned dst_col = r * I4 + i4;
                            const unsigned dst_idx = dst_row * (r3 * I4) + dst_col;
                            
                            const unsigned dst_pack = dst_idx / DstConfig::kMemoryWidthN;
                            const unsigned dst_pos = dst_idx % DstConfig::kMemoryWidthN;
                            
                            if (dst_pos == 0) {
                                next_input[dst_pack] = typename DstConfig::MemoryPackN_t();
                            }
                            next_input[dst_pack][dst_pos] = 
                                gemm_buffers_[CoreIndex].reshape_buffer[src_idx];
                        }
                    }
                }
            }
        }
    }
    else {  // CoreIndex == 3, final output preparation
        // No reshape needed for the final output
        // Just pack the data according to the memory width
        const unsigned total_elements = DstConfig::kGemmN * DstConfig::kGemmM;
        
        #pragma HLS PIPELINE II=1
        for (unsigned i = 0; i < total_elements; ++i) {
            const unsigned pack_idx = i / DstConfig::kMemoryWidthN;
            const unsigned pack_pos = i % DstConfig::kMemoryWidthN;
            
            if (pack_pos == 0) {
                next_input[pack_idx] = typename DstConfig::MemoryPackN_t();
            }
            next_input[pack_idx][pack_pos] = gemm_buffers_[CoreIndex].reshape_buffer[i];
        }
    }
}

template<typename T>
void TTLinear<T>::verify_chain_consistency() const {
    // 验证相邻核心的秩匹配
    if (core0_->next_rank() != core1_->prev_rank() ||
        core1_->next_rank() != core2_->prev_rank() ||
        core2_->next_rank() != core3_->prev_rank()) {
        throw std::runtime_error("Mismatched ranks between adjacent cores");
    }
    
    // 验证首尾秩为1
    if (core0_->prev_rank() != 1 || core3_->next_rank() != 1) {
        throw std::runtime_error("First and last ranks must be 1");
    }
}

template<typename T>
void TTLinear<T>::initialize_random(unsigned seed) {
    core0_->initialize_random(seed++);
    core1_->initialize_random(seed++);
    core2_->initialize_random(seed++);
    core3_->initialize_random(seed++);
}

template<typename T>
void TTLinear<T>::initialize_zeros() {
    core0_->initialize_zeros();
    core1_->initialize_zeros();
    core2_->initialize_zeros();
    core3_->initialize_zeros();
}

template<typename T>
void TTLinear<T>::set_parameters(const std::vector<std::vector<T>>& params) {
    if (params.size() != 4) {
