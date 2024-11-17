#include "tt_linear/tt_linear.h"
#include <numeric>
#include <algorithm>
#include <cassert>

namespace gemm_hls {
namespace tt {

// Config实现部分
template<typename T>
void TTLinear<T>::Config::initialize_default() {
    // 计算每个维度的基本大小
    input_modes.resize(num_cores);
    output_modes.resize(num_cores);
    ranks.resize(num_cores + 1);

    // 计算维度分解
    const unsigned base_input_mode = std::round(std::pow(input_size, 1.0/num_cores));
    const unsigned base_output_mode = std::round(std::pow(output_size, 1.0/num_cores));

    // 设置维度分解
    std::fill(input_modes.begin(), input_modes.end(), base_input_mode);
    std::fill(output_modes.begin(), output_modes.end(), base_output_mode);

    // 设置TT秩
    ranks[0] = ranks[num_cores] = 1;
    std::fill(ranks.begin() + 1, ranks.end() - 1, kMaxRank);

    validate();
}

template<typename T>
void TTLinear<T>::Config::validate() const {
    if (input_modes.size() != num_cores || output_modes.size() != num_cores) {
        throw std::invalid_argument("Incorrect number of modes");
    }
    if (ranks.size() != num_cores + 1) {
        throw std::invalid_argument("Incorrect number of ranks");
    }
    if (!is_valid_decomposition(input_size, input_modes) ||
        !is_valid_decomposition(output_size, output_modes)) {
        throw std::invalid_argument("Invalid dimension decomposition");
    }
    if (ranks.front() != 1 || ranks.back() != 1) {
        throw std::invalid_argument("First and last ranks must be 1");
    }
}

template<typename T>
bool TTLinear<T>::Config::is_valid_decomposition(
    unsigned total, const std::vector<unsigned>& factors) const {
    return total == std::accumulate(
        factors.begin(), factors.end(), 1u, std::multiplies<unsigned>());
}

// TTLinear实现部分
template<typename T>
TTLinear<T>::TTLinear(const Config& config) : config_(config) {
    initialize_cores();
    initialize_buffers();
}

template<typename T>
void TTLinear<T>::initialize_cores() {
    cores_.reserve(config_.num_cores);
    for (unsigned i = 0; i < config_.num_cores; ++i) {
        cores_.emplace_back(
            config_.ranks[i],
            config_.input_modes[i],
            config_.ranks[i + 1],
            config_.output_modes[i]
        );
    }
    verify_chain_consistency();
}

template<typename T>
void TTLinear<T>::initialize_buffers() {
    // 计算最大缓冲区大小
    size_t max_buffer_size = 0;
    for (unsigned i = 0; i < config_.num_cores; ++i) {
        max_buffer_size = std::max(
            max_buffer_size, 
            calculate_buffer_size(i, kMaxBatchSize)
        );
    }

    // 分配缓冲区
    input_buffer_.resize(max_buffer_size);
    output_buffer_.resize(max_buffer_size);
    temp_buffer_.resize(max_buffer_size);
}

template<typename T>
void TTLinear<T>::forward(const T* input, T* output, unsigned batch_size) {
    validate_runtime(input, output, batch_size);
    ensure_buffer_capacity(batch_size);

    // 准备输入数据
    prepare_input(input, batch_size);

    // 依次通过每个TT核心
    const MemoryPackM_t* curr_input = input_buffer_.data();
    MemoryPackM_t* curr_output = output_buffer_.data();

    for (unsigned i = 0; i < config_.num_cores; ++i) {
        compute_single_core(i, curr_input, curr_output, batch_size);

        // 准备下一次迭代
        std::swap(curr_input, curr_output);
        curr_output = (curr_output == output_buffer_.data()) 
                     ? temp_buffer_.data() 
                     : output_buffer_.data();
    }

    // 输出最终结果
    finalize_output(output, batch_size);
}

template<typename T>
void TTLinear<T>::compute_single_core(
    unsigned core_idx,
    const MemoryPackM_t* input,
    MemoryPackM_t* output,
    unsigned batch_size) const {

    // 准备TT核心的矩阵形式
    AlignedVector core_matrix;
    cores_[core_idx].prepare_for_gemm(core_matrix);

    // 获取GEMM维度
    auto dims = get_gemm_dims(core_idx, batch_size);

    // 调用GEMM
    MatrixMultiplicationKernel(
        input,
        core_matrix.data(),
        output,
        dims.M, dims.K, dims.N
    );
}

template<typename T>
void TTLinear<T>::prepare_input(const T* input, unsigned batch_size) const {
    const unsigned elements_per_pack = sizeof(MemoryPackM_t) / sizeof(T);
    const unsigned total_elements = config_.input_size * batch_size;

    for (unsigned i = 0; i < total_elements; i += elements_per_pack) {
        MemoryPackM_t pack;
        for (unsigned j = 0; j < elements_per_pack; ++j) {
            pack[j] = (i + j < total_elements) ? input[i + j] : T(0);
        }
        input_buffer_[i / elements_per_pack] = pack;
    }
}

template<typename T>
void TTLinear<T>::finalize_output(T* output, unsigned batch_size) const {
    const unsigned elements_per_pack = sizeof(MemoryPackM_t) / sizeof(T);
    const unsigned total_elements = config_.output_size * batch_size;

    for (unsigned i = 0; i < total_elements; i += elements_per_pack) {
        const auto& pack = output_buffer_[i / elements_per_pack];
        for (unsigned j = 0; j < elements_per_pack; ++j) {
            if (i + j < total_elements) {
                output[i + j] = pack[j];
            }
        }
    }
}

template<typename T>
typename TTLinear<T>::GemmDimensions TTLinear<T>::get_gemm_dims(
    unsigned core_idx, unsigned batch_size) const {
    
    const auto& core = cores_[core_idx];
    return GemmDimensions{
        .M = batch_size * core.get_gemm_m(),
        .K = core.get_gemm_k(),
        .N = core.get_gemm_n()
    };
}

template<typename T>
size_t TTLinear<T>::calculate_buffer_size(
    unsigned core_idx, unsigned batch_size) const {
    
    auto dims = get_gemm_dims(core_idx, batch_size);
    const unsigned elements_per_pack = sizeof(MemoryPackM_t) / sizeof(T);
    return ((dims.M * dims.N + elements_per_pack - 1) / elements_per_pack);
}

template<typename T>
void TTLinear<T>::validate_runtime(
    const T* input, T* output, unsigned batch_size) const {
    
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("Null input/output pointers");
    }
    if (batch_size == 0 || batch_size > kMaxBatchSize) {
        throw std::invalid_argument("Invalid batch size");
    }
}

template<typename T>
void TTLinear<T>::verify_chain_consistency() const {
    validate_tt_core_chain(cores_);
}

template<typename T>
void TTLinear<T>::ensure_buffer_capacity(unsigned batch_size) const {
    size_t required_size = 0;
    for (unsigned i = 0; i < config_.num_cores; ++i) {
        required_size = std::max(
            required_size,
            calculate_buffer_size(i, batch_size)
        );
    }

    if (required_size > input_buffer_.size()) {
        throw std::runtime_error("Buffer capacity exceeded");
    }
}

template<typename T>
void TTLinear<T>::print_dimensions() const {
    std::cout << "TT-Linear layer dimensions:\n"
              << "  Input size: " << config_.input_size << "\n"
              << "  Output size: " << config_.output_size << "\n"
              << "  Number of cores: " << config_.num_cores << "\n"
              << "  Input modes: ";
    for (auto m : config_.input_modes) std::cout << m << " ";
    std::cout << "\n  Output modes: ";
    for (auto m : config_.output_modes) std::cout << m << " ";
    std::cout << "\n  Ranks: ";
    for (auto r : config_.ranks) std::cout << r << " ";
    std::cout << std::endl;
}

// 显式实例化
template class TTLinear<float>;
template class TTLinear<half>;

} // namespace tt
} // namespace gemm_hls
