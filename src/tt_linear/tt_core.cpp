#include "tt_linear/tt_core.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>

namespace gemm_hls {
namespace tt {

template<typename T>
TTCore<T>::TTCore(unsigned rank1, unsigned in_dim, unsigned rank2, unsigned out_dim)
    : rank1_(rank1), in_dim_(in_dim), rank2_(rank2), out_dim_(out_dim) {
    
    // 验证维度
    if (!validate_dimensions()) {
        throw std::invalid_argument("Invalid TT core dimensions");
    }
    
    // 分配对齐的内存
    const size_t size = compute_aligned_size();
    data_.resize(size);
    
    // 初始化为0
    initialize_zeros();
}

template<typename T>
size_t TTCore<T>::compute_aligned_size() const {
    // 计算实际需要的元素数量
    size_t raw_size = rank1_ * in_dim_ * rank2_ * out_dim_;
    // 计算对齐所需的额外元素
    size_t align_elements = kMemoryAlignment / sizeof(T);
    // 向上取整到对齐边界
    return ((raw_size + align_elements - 1) / align_elements) * align_elements;
}

template<typename T>
void TTCore<T>::prepare_for_gemm(
    std::vector<MemoryPackM_t, AlignedAllocator<MemoryPackM_t, kMemoryAlignment>>& matrix) const {
    
    const unsigned rows = get_gemm_m();
    const unsigned cols = get_gemm_k() * get_gemm_n();
    
    // 计算需要的MemoryPack数量
    const unsigned elements_per_pack = sizeof(MemoryPackM_t) / sizeof(T);
    const unsigned num_packs = (rows * cols + elements_per_pack - 1) / elements_per_pack;
    matrix.resize(num_packs);
    
    // 重组数据为GEMM格式
    #pragma HLS PIPELINE
    for(unsigned i1 = 0; i1 < rank1_; ++i1) {
        for(unsigned i2 = 0; i2 < in_dim_; ++i2) {
            for(unsigned i3 = 0; i3 < rank2_; ++i3) {
                for(unsigned i4 = 0; i4 < out_dim_; ++i4) {
                    // 计算源和目标索引
                    const unsigned src_idx = compute_flat_index(i1, i2, i3, i4);
                    const unsigned row = i1 * in_dim_ + i2;
                    const unsigned col = i3 * out_dim_ + i4;
                    const unsigned dst_idx = row * cols + col;
                    
                    // 写入到内存对齐的格式
                    const unsigned pack_idx = dst_idx / elements_per_pack;
                    const unsigned pack_offset = dst_idx % elements_per_pack;
                    matrix[pack_idx][pack_offset] = data_[src_idx];
                }
            }
        }
    }
}

template<typename T>
void TTCore<T>::initialize_random(unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(float(in_dim_)));
    
    const size_t actual_size = rank1_ * in_dim_ * rank2_ * out_dim_;
    for(size_t i = 0; i < actual_size; ++i) {
        if constexpr (std::is_same_v<T, half>) {
            data_[i] = static_cast<T>(dist(gen));
        } else {
            data_[i] = static_cast<T>(dist(gen));
        }
    }
    
    // 填充对齐部分为0
    std::fill(data_.begin() + actual_size, data_.end(), T(0));
}

template<typename T>
void TTCore<T>::initialize_zeros() {
    std::fill(data_.begin(), data_.end(), T(0));
}

template<typename T>
void TTCore<T>::set_data(const std::vector<T>& new_data) {
    const size_t required_size = rank1_ * in_dim_ * rank2_ * out_dim_;
    if(new_data.size() != required_size) {
        throw std::invalid_argument("Invalid data size for TT core");
    }
    
    // 复制数据
    std::copy(new_data.begin(), new_data.end(), data_.begin());
    // 确保对齐部分为0
    std::fill(data_.begin() + required_size, data_.end(), T(0));
}

template<typename T>
bool TTCore<T>::validate_dimensions() const {
    // 基本维度检查
    if (rank1_ == 0 || in_dim_ == 0 || rank2_ == 0 || out_dim_ == 0) {
        return false;
    }
    
    // 大小限制检查
    if (rank1_ > kMaxRank || rank2_ > kMaxRank) {
        return false;
    }
    
    // 验证维度与GEMM约束的兼容性
    if ((rank1_ * in_dim_) % kComputeTileSizeM != 0 ||
        (rank2_ * out_dim_) % kComputeTileSizeN != 0) {
        return false;
    }
    
    return true;
}

template<typename T>
void TTCore<T>::print_dimensions() const {
    std::cout << "TT Core dimensions:\n"
              << "  rank1: " << rank1_ << "\n"
              << "  in_dim: " << in_dim_ << "\n"
              << "  rank2: " << rank2_ << "\n"
              << "  out_dim: " << out_dim_ << "\n"
              << "  raw size: " << raw_size() << "\n"
              << "  aligned size: " << size() << std::endl;
}

// 辅助函数实现
template<typename T>
void validate_tt_core_chain(const std::vector<TTCore<T>>& cores) {
    if(cores.empty()) {
        throw std::invalid_argument("Empty TT core chain");
    }
    
    // 验证相邻核心的秩匹配
    for(size_t i = 0; i < cores.size() - 1; ++i) {
        if(cores[i].rank2() != cores[i + 1].rank1()) {
            throw std::invalid_argument("Mismatched ranks between adjacent cores");
        }
    }
    
    // 验证首尾秩为1
    if(cores.front().rank1() != 1 || cores.back().rank2() != 1) {
        throw std::invalid_argument("First and last ranks must be 1");
    }
}

template<typename T>
void print_tt_core_info(const TTCore<T>& core) {
    core.print_dimensions();
    std::cout << "GEMM dimensions:\n"
              << "  M: " << core.get_gemm_m() << "\n"
              << "  K: " << core.get_gemm_k() << "\n"
              << "  N: " << core.get_gemm_n() << std::endl;
}

// 显式实例化
template class TTCore<float>;
template class TTCore<half>;
template void validate_tt_core_chain<float>(const std::vector<TTCore<float>>&);
template void validate_tt_core_chain<half>(const std::vector<TTCore<half>>&);
template void print_tt_core_info<float>(const TTCore<float>&);
template void print_tt_core_info<half>(const TTCore<half>&);

} // namespace tt
} // namespace gemm_hls
