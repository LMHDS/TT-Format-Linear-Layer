#pragma once
#include "Config.h"
#include "MatrixMultiplication.h"
#include <vector>
#include <memory>
#include <cassert>
#include <type_traits>

namespace gemm_hls {
namespace tt {

// AlignedAllocator保持不变
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

// 每个TT核心的GEMM维度定义
struct TTCoreDimensions {
    struct FirstCore {
        unsigned N;    // I2*I3*I4
        unsigned K;    // I1
        unsigned M;    // r1*O1
    };

    struct SecondCore {
        unsigned N;    // O1*I3*I4
        unsigned K;    // r1*I2
        unsigned M;    // r2*O2
    };

    struct ThirdCore {
        unsigned N;    // O1*O2*I4
        unsigned K;    // r2*I3
        unsigned M;    // r3*O3
    };

    struct FourthCore {
        unsigned N;    // O1*O2*O3
        unsigned K;    // r3*I4
        unsigned M;    // r4*O4
    };
};

template<typename T = Data_t, int CoreIndex = 0>
class TTCore {
public:
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                 "TTCore only supports float and half data types");
    
    static_assert(CoreIndex >= 0 && CoreIndex < kTTNumCores,
                 "Invalid CoreIndex");

    // 使用对应核心的配置
    using CoreConfig = tt_core##CoreIndex;
    
    // 定义各种内存包类型
    using MemoryPackN_t = hlslib::DataPack<T, CoreConfig::kMemoryWidthN>;
    using MemoryPackK_t = hlslib::DataPack<T, CoreConfig::kMemoryWidthK>;
    using MemoryPackM_t = hlslib::DataPack<T, CoreConfig::kMemoryWidthM>;
    
    using ComputePackN_t = hlslib::DataPack<T, CoreConfig::kComputeTileSizeN>;
    using ComputePackM_t = hlslib::DataPack<T, CoreConfig::kComputeTileSizeM>;
    
    // TT Core constructor
    TTCore(unsigned prev_rank,    // r_{i-1}
           unsigned in_mode,      // m_i
           unsigned next_rank,    // r_i
           unsigned out_mode);    // n_i
    
    // 移动语义
    TTCore(const TTCore&) = delete;
    TTCore& operator=(const TTCore&) = delete;
    TTCore(TTCore&&) = default;
    TTCore& operator=(TTCore&&) = default;

    // 基本维度访问
    unsigned prev_rank() const { return prev_rank_; }  // r_{i-1}
    unsigned in_mode() const { return in_mode_; }      // m_i
    unsigned next_rank() const { return next_rank_; }  // r_i
    unsigned out_mode() const { return out_mode_; }    // n_i

    // 核心信息
    static constexpr int core_index() { return CoreIndex; }
    static constexpr bool is_first_core() { return CoreIndex == 0; }
    static constexpr bool is_last_core() { return CoreIndex == kTTNumCores - 1; }

    // 数据访问
    size_t size() const { return data_.size(); }
    const T* data() const { return data_.data(); }
    T* data() { return data_.data(); }

    // GEMM数据准备
    void prepare_for_gemm(std::vector<MemoryPackM_t, 
                         AlignedAllocator<MemoryPackM_t, CoreConfig::kMemoryAlignment>>& matrix) const;

    // 获取GEMM维度
    struct GemmDimensions {
        unsigned N, K, M;
    };
    
    GemmDimensions get_gemm_dims() const {
        return {
            CoreConfig::kGemmN,
            CoreConfig::kGemmK,
            CoreConfig::kGemmM
        };
    }

    // 数据操作
    void initialize_random(unsigned seed = 42);
    void initialize_zeros();
    void set_data(const std::vector<T>& new_data);

    // Memory pack类型访问
    using MemoryPackType = MemoryPackM_t;
    static constexpr unsigned kMemoryWidth = CoreConfig::kMemoryWidthM;
    static constexpr unsigned kComputeTileSize = CoreConfig::kComputeTileSizeM;

    // Width conversion checks
    static constexpr bool needs_width_conversion_N() {
        return CoreConfig::kGemmN % CoreConfig::kMemoryWidthN != 0;
    }
    
    static constexpr bool needs_width_conversion_K() {
        return CoreConfig::kGemmK % CoreConfig::kMemoryWidthK != 0;
    }
    
    static constexpr bool needs_width_conversion_M() {
        return CoreConfig::kGemmM % CoreConfig::kMemoryWidthM != 0;
    }

    // Debug helpers
    void print_dimensions() const;

private:
    unsigned prev_rank_;  // r_{i-1}
    unsigned in_mode_;    // m_i
    unsigned next_rank_;  // r_i
    unsigned out_mode_;   // n_i

    std::vector<T, AlignedAllocator<T, CoreConfig::kMemoryAlignment>> data_;

    // 数据布局验证
    void validate_dimensions() const {
        if (prev_rank_ * in_mode_ * next_rank_ * out_mode_ != 
            CoreConfig::kGemmN * CoreConfig::kGemmK * CoreConfig::kGemmM) {
            throw std::invalid_argument("Dimensions do not match GEMM configuration");
        }
        validate_width_requirements();
    }

    void validate_width_requirements() const {
        static_assert(CoreConfig::kMemoryWidthN > 0, "Invalid memory width N");
        static_assert(CoreConfig::kMemoryWidthK > 0, "Invalid memory width K");
        static_assert(CoreConfig::kMemoryWidthM > 0, "Invalid memory width M");
        
        if (needs_width_conversion_N() || needs_width_conversion_K() || needs_width_conversion_M()) {
            #ifndef MM_CONVERT_WIDTH
            throw std::runtime_error("Width conversion needed but not enabled");
            #endif
        }
    }

    bool validate_config_compatibility() const;
    size_t compute_aligned_size() const;
    unsigned compute_flat_index(unsigned i1, unsigned i2, unsigned i3, unsigned i4) const;

    // Width conversion helpers
    void convert_width_for_gemm(const T* input, std::vector<MemoryPackM_t>& output) const;
    void convert_width_from_gemm(const std::vector<MemoryPackM_t>& input, T* output) const;
};

// 辅助函数
template<typename T>
void validate_tt_core_chain(const std::vector<std::unique_ptr<TTCore<T>>>& cores);

template<typename T, int CoreIndex>
void print_tt_core_info(const TTCore<T, CoreIndex>& core);

// 显式实例化声明
extern template class TTCore<float, 0>;
extern template class TTCore<float, 1>;
extern template class TTCore<float, 2>;
extern template class TTCore<float, 3>;

extern template class TTCore<half, 0>;
extern template class TTCore<half, 1>;
extern template class TTCore<half, 2>;
extern template class TTCore<half, 3>;

} // namespace tt
} // namespace gemm_hls
