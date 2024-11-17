#pragma once

#include "Config.h"
#include "MatrixMultiplication.h"
#include <vector>
#include <memory>
#include <cassert>

namespace gemm_hls {
namespace tt {

// 对齐内存分配器
template<typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    static constexpr std::size_t alignment = Alignment;

    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T))) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) {
        free(p);
    }
};

template<typename T = TTDataType>
class TTCore {
public:
    // 构造函数：指定核心的维度
    TTCore(unsigned rank1, unsigned in_dim, unsigned rank2, unsigned out_dim);
    
    // 删除拷贝构造和赋值，允许移动
    TTCore(const TTCore&) = delete;
    TTCore& operator=(const TTCore&) = delete;
    TTCore(TTCore&&) = default;
    TTCore& operator=(TTCore&&) = default;

    // 基本维度访问
    unsigned rank1() const { return rank1_; }
    unsigned in_dim() const { return in_dim_; }
    unsigned rank2() const { return rank2_; }
    unsigned out_dim() const { return out_dim_; }

    // 获取数据大小
    size_t size() const { return data_.size(); }
    size_t raw_size() const { return rank1_ * in_dim_ * rank2_ * out_dim_; }

    // 数据访问
    const T* data() const { return data_.data(); }
    T* data() { return data_.data(); }

    // GEMM相关接口
    void prepare_for_gemm(std::vector<MemoryPackM_t, AlignedAllocator<MemoryPackM_t, kMemoryAlignment>>& matrix) const;
    
    // GEMM维度信息
    unsigned get_gemm_m() const { return rank1_ * in_dim_; }
    unsigned get_gemm_k() const { return rank2_; }
    unsigned get_gemm_n() const { return out_dim_; }

    // 数据操作
    void initialize_random(unsigned seed = 42);
    void initialize_zeros();
    void set_data(const std::vector<T>& new_data);

    // 调试辅助
    void print_dimensions() const;
    bool validate_dimensions() const;

private:
    unsigned rank1_;   // 输入TT秩
    unsigned in_dim_;  // 输入维度
    unsigned rank2_;   // 输出TT秩
    unsigned out_dim_; // 输出维度

    // 使用对齐的内存存储
    std::vector<T, AlignedAllocator<T, kMemoryAlignment>> data_;

    // 索引计算
    unsigned compute_flat_index(unsigned i1, unsigned i2, unsigned i3, unsigned i4) const {
        return ((i1 * in_dim_ + i2) * rank2_ + i3) * out_dim_ + i4;
    }

    // 数据布局验证
    void validate_data_layout() const;

    // 计算所需的总大小（包含对齐）
    size_t compute_aligned_size() const;
};

// 辅助函数声明
template<typename T>
void validate_tt_core_chain(const std::vector<TTCore<T>>& cores);

template<typename T>
void print_tt_core_info(const TTCore<T>& core);

} // namespace tt
} // namespace gemm_hls
