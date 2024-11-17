#pragma once

#include "Config.h"
#include "tt_core.h"
#include "MatrixMultiplication.h"
#include <vector>
#include <memory>
#include <cassert>

namespace gemm_hls {
namespace tt {

template<typename T = TTDataType>
class TTLinear {
public:
    struct Config {
        const unsigned input_size;   // 总输入维度
        const unsigned output_size;  // 总输出维度
        const unsigned num_cores;    // TT核心数量
        std::vector<unsigned> input_modes;  // 输入维度分解 [m1,m2,...,md]
        std::vector<unsigned> output_modes; // 输出维度分解 [n1,n2,...,nd]
        std::vector<unsigned> ranks;        // TT秩 [r0,r1,...,rd]

        // 默认构造函数 - 使用配置文件中的值
        Config() 
            : input_size(kInputSize),
              output_size(kOutputSize),
              num_cores(kNumCores) {
            initialize_default();
        }

        // 自定义构造函数
        Config(std::vector<unsigned> in_modes,
               std::vector<unsigned> out_modes,
               std::vector<unsigned> tt_ranks)
            : input_size(kInputSize),
              output_size(kOutputSize),
              num_cores(kNumCores),
              input_modes(std::move(in_modes)),
              output_modes(std::move(out_modes)),
              ranks(std::move(tt_ranks)) {
            validate();
        }

    private:
        void initialize_default();
        void validate() const;
        bool is_valid_decomposition(unsigned total, const std::vector<unsigned>& factors) const;
    };

    // 构造函数
    explicit TTLinear(const Config& config = Config());
    
    // 禁用拷贝，允许移动
    TTLinear(const TTLinear&) = delete;
    TTLinear& operator=(const TTLinear&) = delete;
    TTLinear(TTLinear&&) = default;
    TTLinear& operator=(TTLinear&&) = default;

    // 核心功能：前向计算
    void forward(const T* input, T* output, unsigned batch_size = 1);

    // 参数访问和设置
    void set_parameters(const std::vector<std::vector<T>>& params);
    std::vector<std::vector<T>> get_parameters() const;

    // 维度信息
    unsigned input_size() const { return config_.input_size; }
    unsigned output_size() const { return config_.output_size; }
    unsigned num_cores() const { return config_.num_cores; }

private:
    Config config_;
    std::vector<TTCore<T>> cores_;

    // GEMM计算相关的内存缓冲区
    using AlignedVector = std::vector<MemoryPackM_t, 
          AlignedAllocator<MemoryPackM_t, kMemoryAlignment>>;
    
    mutable AlignedVector input_buffer_;   // 输入缓冲
    mutable AlignedVector output_buffer_;  // 输出缓冲
    mutable AlignedVector temp_buffer_;    // 临时计算缓冲

    // 内部计算方法
    void compute_single_core(
        unsigned core_idx,
        const MemoryPackM_t* input,
        MemoryPackM_t* output,
        unsigned batch_size) const;

    // 数据转换方法
    void prepare_input(const T* input, unsigned batch_size) const;
    void finalize_output(T* output, unsigned batch_size) const;

    // 辅助函数
    void initialize_cores();
    void initialize_buffers();
    void validate_runtime(const T* input, T* output, unsigned batch_size) const;

    // GEMM维度计算
    struct GemmDimensions {
        unsigned M, K, N;
    };
    GemmDimensions get_gemm_dims(unsigned core_idx, unsigned batch_size) const;

    // 缓冲区管理
    size_t calculate_buffer_size(unsigned core_idx, unsigned batch_size) const;
    void ensure_buffer_capacity(unsigned batch_size) const;

    // 调试辅助
    void print_dimensions() const;
    void verify_chain_consistency() const;
};

// 调试和验证辅助函数
template<typename T>
void print_tt_linear_info(const TTLinear<T>& layer);

template<typename T>
bool verify_tt_dimensions(const typename TTLinear<T>::Config& config);

} // namespace tt
} // namespace gemm_hls
