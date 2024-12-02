#pragma once
#include "Config.h"
#include "tt_core.h"
#include "MatrixMultiplication.h"
#include <vector>
#include <memory>
#include <cassert>

namespace gemm_hls {
namespace tt {

template<typename T = Data_t>
class TTLinear {
public:
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                 "TTLinear only supports float and half data types");

    struct Config {
        Config() 
            : input_size(kTTInputSize),
              output_size(kTTOutputSize),
              num_cores(kTTNumCores) {
            initialize_default();
        }

        Config(std::vector<unsigned> in_modes,
               std::vector<unsigned> out_modes,
               std::vector<unsigned> tt_ranks)
            : input_size(kTTInputSize),
              output_size(kTTOutputSize),
              num_cores(kTTNumCores),
              input_modes(std::move(in_modes)),
              output_modes(std::move(out_modes)),
              ranks(std::move(tt_ranks)) {
            validate();
        }

        const unsigned input_size;    // I1*I2*I3*I4
        const unsigned output_size;   // O1*O2*O3*O4
        const unsigned num_cores;     // 固定为4
        std::vector<unsigned> input_modes;   // [I1,I2,I3,I4]
        std::vector<unsigned> output_modes;  // [O1,O2,O3,O4]
        std::vector<unsigned> ranks;         // [r0,r1,r2,r3,r4]

    private:
        void initialize_default();
        void validate() const;
        bool is_valid_decomposition(unsigned total, const std::vector<unsigned>& factors) const;
    };

    explicit TTLinear(const Config& config = Config());
    
    TTLinear(const TTLinear&) = delete;
    TTLinear& operator=(const TTLinear&) = delete;
    TTLinear(TTLinear&&) = default;
    TTLinear& operator=(TTLinear&&) = default;

    void forward(const T* input, T* output);
    void set_parameters(const std::vector<std::vector<T>>& params);
    std::vector<std::vector<T>> get_parameters() const;

    unsigned input_size() const { return config_.input_size; }
    unsigned output_size() const { return config_.output_size; }
    unsigned num_cores() const { return config_.num_cores; }

    void initialize_random(unsigned seed = 42);
    void initialize_zeros();
    void print_dimensions() const;

private:
    Config config_;
    // 使用明确的核心类型
    std::vector<std::unique_ptr<TTCore<T, 0>>> core0_;
    std::vector<std::unique_ptr<TTCore<T, 1>>> core1_;
    std::vector<std::unique_ptr<TTCore<T, 2>>> core2_;
    std::vector<std::unique_ptr<TTCore<T, 3>>> core3_;

    // 为每个核心定义所需的类型
    template<int CoreIndex>
    using CoreConfig = typename TTCore<T, CoreIndex>::CoreConfig;
    
    template<int CoreIndex>
    using MemoryPackN_t = typename TTCore<T, CoreIndex>::MemoryPackN_t;
    
    template<int CoreIndex>
    using MemoryPackK_t = typename TTCore<T, CoreIndex>::MemoryPackK_t;
    
    template<int CoreIndex>
    using MemoryPackM_t = typename TTCore<T, CoreIndex>::MemoryPackM_t;

    template<int CoreIndex>
    using AlignedVector = std::vector<MemoryPackM_t<CoreIndex>, 
                                    AlignedAllocator<MemoryPackM_t<CoreIndex>, 
                                                   CoreConfig<CoreIndex>::kMemoryAlignment>>;

    // 每个核心的缓冲区
    mutable AlignedVector<0> buffer0_input_;
    mutable AlignedVector<0> buffer0_output_;
    mutable AlignedVector<1> buffer1_input_;
    mutable AlignedVector<1> buffer1_output_;
    mutable AlignedVector<2> buffer2_input_;
    mutable AlignedVector<2> buffer2_output_;
    mutable AlignedVector<3> buffer3_input_;
    mutable AlignedVector<3> buffer3_output_;

    // 数据准备函数 - 每个核心有自己的实现
    template<int CoreIndex>
    void prepare_gemm_input(const T* input, AlignedVector<CoreIndex>& matrix) const;

    template<int CoreIndex>
    void prepare_gemm_output(const AlignedVector<CoreIndex>& matrix, T* output) const;

    // 每个核心的计算和数据重组函数
    void compute_first_core(const T* input, T* output) const {
        using Config = CoreConfig<0>;
        prepare_gemm_input<0>(input, buffer0_input_);
        cores_[0]->prepare_for_gemm(buffer0_output_);
        
        MatrixMultiplicationKernel(
            buffer0_input_.data(),
            buffer0_output_.data(),
            buffer0_output_.data(),
            Config::kGemmN,
            Config::kGemmK,
            Config::kGemmM
        );

        prepare_gemm_output<0>(buffer0_output_, output);
    }

    void compute_second_core(const T* input, T* output) const;
    void compute_third_core(const T* input, T* output) const;
    void compute_fourth_core(const T* input, T* output) const;

    // 数据重组函数
    template<int CoreIndex>
    void reshape_for_next_core(const T* input, T* output) const;

    // 维度计算函数 - 使用核心配置
    template<int CoreIndex>
    auto get_core_dimensions() const {
        return TTCore<T, CoreIndex>::CoreConfig::get_gemm_dims();
    }

    // 辅助方法
    void initialize_cores();
    void initialize_buffers();
    void verify_chain_consistency() const;
};

// 显式实例化声明
extern template class TTLinear<float>;
extern template class TTLinear<half>;

} // namespace tt
} // namespace gemm_hls
