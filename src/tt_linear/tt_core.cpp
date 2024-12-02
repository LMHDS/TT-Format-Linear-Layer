#include "tt_linear/tt_core.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>

namespace gemm_hls {
namespace tt {

template<typename T, int CoreIndex>
TTCore<T, CoreIndex>::TTCore(unsigned prev_rank, unsigned in_mode, 
                            unsigned next_rank, unsigned out_mode)
    : prev_rank_(prev_rank), in_mode_(in_mode), 
      next_rank_(next_rank), out_mode_(out_mode) {
    
    validate_dimensions();
    const size_t size = compute_aligned_size();
    data_.resize(size);
    initialize_zeros();
}

template<typename T, int CoreIndex>
void TTCore<T, CoreIndex>::prepare_for_gemm(
    std::vector<MemoryPackM_t, AlignedAllocator<MemoryPackM_t, CoreConfig::kMemoryAlignment>>& matrix) const {
    
    auto dims = get_gemm_dims();
    
    // 第一步：转换为TT格式的MemoryPack格式
    const unsigned elements_per_pack = sizeof(MemoryPackM_t) / sizeof(T);
    const unsigned num_packs = (dims.N * dims.K + elements_per_pack - 1) / elements_per_pack;
    matrix.resize(num_packs);

    // 根据核心配置获取memory width
    constexpr unsigned mem_width = CoreConfig::kMemoryWidthM;

    // 重组数据为GEMM所需的Memory格式
    #pragma HLS PIPELINE II=1
    for(unsigned n = 0; n < dims.N; ++n) {
        for(unsigned k = 0; k < dims.K; ++k) {
            const unsigned dst_idx = n * dims.K + k;
            const unsigned pack_idx = dst_idx / mem_width;
            const unsigned pack_offset = dst_idx % mem_width;
            
            // 计算TT格式源索引
            unsigned src_idx;
            if constexpr (CoreIndex == 0) {
                src_idx = compute_flat_index(1, n, k / out_mode_, k % out_mode_);
            }
            else {
                src_idx = compute_flat_index(n / in_mode_, n % in_mode_, 
                                         k / out_mode_, k % out_mode_);
            }

            if (pack_offset == 0) {
                matrix[pack_idx] = MemoryPackM_t();
            }
            matrix[pack_idx][pack_offset] = data_[src_idx];
        }
    }

    // 第二步：对需要的核心进行宽度转换
    if constexpr (CoreIndex == 0) {
        #ifdef TT_CORE0_CONVERT_WIDTH
        do_width_conversion(matrix, dims);
        #endif
    }
    else if constexpr (CoreIndex == 1) {
        #ifdef TT_CORE1_CONVERT_WIDTH
        do_width_conversion(matrix, dims);
        #endif
    }
    else if constexpr (CoreIndex == 2) {
        #ifdef TT_CORE2_CONVERT_WIDTH
        do_width_conversion(matrix, dims);
        #endif
    }
    else if constexpr (CoreIndex == 3) {
        #ifdef TT_CORE3_CONVERT_WIDTH
        do_width_conversion(matrix, dims);
        #endif
    }
}

template<typename T, int CoreIndex>
void TTCore<T, CoreIndex>::do_width_conversion(
    std::vector<MemoryPackM_t>& matrix,
    const GemmDimensions& dims) const {
    
    Stream<MemoryPackM_t> wide_stream("wide_stream");
    Stream<ComputePackM_t> narrow_stream("narrow_stream");
    std::vector<ComputePackM_t> converted;

    // Push to wide stream
    for (const auto& pack : matrix) {
        wide_stream.Push(pack);
    }

    // Convert width using Memory module function
    ConvertWidthB(wide_stream, narrow_stream, dims.N, dims.K, dims.M);

    // Collect converted data
    const unsigned total_reads = dims.N * dims.K;
    converted.resize(total_reads / CoreConfig::kComputeTileSizeM);
    for (unsigned i = 0; i < converted.size(); ++i) {
        converted[i] = narrow_stream.Pop();
    }

    // Update matrix
    matrix.clear();
    matrix.resize(converted.size());
    for (unsigned i = 0; i < converted.size(); ++i) {
        matrix[i] = converted[i];
    }
}

template<typename T, int CoreIndex>
size_t TTCore<T, CoreIndex>::compute_aligned_size() const {
    size_t raw_size = prev_rank_ * in_mode_ * next_rank_ * out_mode_;
    size_t align_elements = CoreConfig::kMemoryAlignment / sizeof(T);
    return ((raw_size + align_elements - 1) / align_elements) * align_elements;
}

template<typename T, int CoreIndex>
unsigned TTCore<T, CoreIndex>::compute_flat_index(
    unsigned i1, unsigned i2, unsigned i3, unsigned i4) const {
    
    if constexpr (CoreIndex == 0) {
        return i1 * (in_mode_ * next_rank_ * out_mode_) +
               i2 * (next_rank_ * out_mode_) +
               i3 * out_mode_ + i4;
    }
    else {
        return i1 * (in_mode_ * next_rank_ * out_mode_) +
               i2 * (next_rank_ * out_mode_) +
               i3 * out_mode_ + i4;
    }
}

template<typename T, int CoreIndex>
void TTCore<T, CoreIndex>::initialize_random(unsigned seed) {
    std::mt19937 gen(seed);
    float scale = 1.0f / std::sqrt(float(in_mode_ * prev_rank_));
    std::normal_distribution<float> dist(0.0f, scale);
    
    const size_t actual_size = prev_rank_ * in_mode_ * next_rank_ * out_mode_;
    for(size_t i = 0; i < actual_size; ++i) {
        if constexpr (std::is_same_v<T, half>) {
            data_[i] = static_cast<T>(__float2half(dist(gen)));
        } else {
            data_[i] = static_cast<T>(dist(gen));
        }
    }
    
    std::fill(data_.begin() + actual_size, data_.end(), T(0));
}

template<typename T, int CoreIndex>
void TTCore<T, CoreIndex>::initialize_zeros() {
    std::fill(data_.begin(), data_.end(), T(0));
}

template<typename T, int CoreIndex>
void TTCore<T, CoreIndex>::set_data(const std::vector<T>& new_data) {
    const size_t required_size = prev_rank_ * in_mode_ * next_rank_ * out_mode_;
    if(new_data.size() != required_size) {
        throw std::invalid_argument("Invalid data size for TT core");
    }
    
    std::copy(new_data.begin(), new_data.end(), data_.begin());
    std::fill(data_.begin() + required_size, data_.end(), T(0));
}

template<typename T, int CoreIndex>
void TTCore<T, CoreIndex>::print_dimensions() const {
    std::cout << "TT Core " << CoreIndex << " dimensions:\n"
              << "  prev_rank (r_{i-1}): " << prev_rank_ << "\n"
              << "  in_mode (m_i): " << in_mode_ << "\n"
              << "  next_rank (r_i): " << next_rank_ << "\n"
              << "  out_mode (n_i): " << out_mode_ << "\n"
              << "  aligned size: " << size() << "\n"
              << "GEMM dimensions:\n"
              << "  N: " << CoreConfig::kGemmN << "\n"
              << "  K: " << CoreConfig::kGemmK << "\n"
              << "  M: " << CoreConfig::kGemmM << "\n"
              << "Memory widths:\n"
              << "  N: " << CoreConfig::kMemoryWidthN << "\n"
              << "  K: " << CoreConfig::kMemoryWidthK << "\n"
              << "  M: " << CoreConfig::kMemoryWidthM << std::endl;
}

template<typename T>
void validate_tt_core_chain(const std::vector<std::unique_ptr<TTCore<T>>>& cores) {
    if (cores.empty()) {
        throw std::invalid_argument("Empty TT core chain");
    }
    
    for (size_t i = 0; i < cores.size() - 1; ++i) {
        if (cores[i]->next_rank() != cores[i + 1]->prev_rank()) {
            throw std::invalid_argument("Mismatched ranks between adjacent cores");
        }
    }
    
    if (cores.front()->prev_rank() != 1 || cores.back()->next_rank() != 1) {
        throw std::invalid_argument("First and last ranks must be 1");
    }
}

// 显式实例化
template class TTCore<float, 0>;
template class TTCore<float, 1>;
template class TTCore<float, 2>;
template class TTCore<float, 3>;

template class TTCore<half, 0>;
template class TTCore<half, 1>;
template class TTCore<half, 2>;
template class TTCore<half, 3>;

} // namespace tt
} // namespace gemm_hls
