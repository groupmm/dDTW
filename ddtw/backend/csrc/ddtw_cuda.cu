#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <limits>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INT32(x) TORCH_CHECK(x.scalar_type() == at::kInt, #x " must be int32")

namespace {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_STEPS = 32;
constexpr int MINFUNC_SOFTMIN = 1;
constexpr int MINFUNC_SPARSEMIN = 2;
constexpr int MINFUNC_SMOOTHMIN = 3;
constexpr int MINFUNC_HARDMIN = 4;

inline int ceil_div(int64_t a, int64_t b) {
    return static_cast<int>((a + b - 1) / b);
}

__device__ __forceinline__ int idx3(int b, int n, int m, int N, int M) {
    return (b * N + n) * M + m;
}

__device__ __forceinline__ int idx4(int b, int n, int m, int s, int N, int M, int S) {
    return ((b * N + n) * M + m) * S + s;
}

__device__ void softmin_device(const float* d_dir, int n_steps, float gamma, float* val, float* grads) {
    bool all_inf = true;
    for (int i = 0; i < n_steps; ++i) {
        all_inf = all_inf && isinf(d_dir[i]);
    }

    if (all_inf) {
        *val = INFINITY;
        float g = 1.0f / static_cast<float>(n_steps);
        for (int i = 0; i < n_steps; ++i) {
            grads[i] = g;
        }
        return;
    }

    float min_d = INFINITY;
    for (int i = 0; i < n_steps; ++i) {
        min_d = fminf(min_d, d_dir[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < n_steps; ++i) {
        sum_exp += expf(-(d_dir[i] - min_d) / gamma);
    }

    *val = -gamma * (-min_d / gamma + logf(sum_exp));
    for (int i = 0; i < n_steps; ++i) {
        grads[i] = expf(-(d_dir[i] - min_d) / gamma) / sum_exp;
    }
}

__device__ void hardmin_device(const float* d_dir, int n_steps, float* val, float* grads) {
    bool all_inf = true;
    for (int i = 0; i < n_steps; ++i) {
        all_inf = all_inf && isinf(d_dir[i]);
        grads[i] = 0.0f;
    }

    if (all_inf) {
        *val = INFINITY;
        float g = 1.0f / static_cast<float>(n_steps);
        for (int i = 0; i < n_steps; ++i) {
            grads[i] = g;
        }
        return;
    }

    int min_index = 0;
    float min_val = d_dir[0];
    for (int i = 0; i < n_steps; ++i) {
        if (d_dir[i] < min_val) {
            min_val = d_dir[i];
            min_index = i;
        }
    }
    *val = min_val;
    grads[min_index] = 1.0f;
}

__device__ void sparsemin_device(const float* r, int n_features, float gamma, float* val, float* grads) {
    float local_x[MAX_STEPS];
    float local_u[MAX_STEPS];
    float local_cssv[MAX_STEPS];

    bool all_inf = true;
    for (int i = 0; i < n_features; ++i) {
        all_inf = all_inf && (r[i] > 1e10f);
    }
    if (all_inf) {
        *val = INFINITY;
        float g = 1.0f / static_cast<float>(n_features);
        for (int i = 0; i < n_features; ++i) {
            grads[i] = g;
        }
        return;
    }

    for (int i = 0; i < n_features; ++i) {
        local_x[i] = (r[i] > 1e20f ? 1e20f : r[i]);
        local_x[i] = -local_x[i] / gamma;
        local_u[i] = local_x[i];
    }

    for (int i = 1; i < n_features; ++i) {
        float key = local_u[i];
        int j = i - 1;
        while (j >= 0 && local_u[j] < key) {
            local_u[j + 1] = local_u[j];
            --j;
        }
        local_u[j + 1] = key;
    }

    local_cssv[0] = local_u[0] - 1.0f;
    for (int i = 1; i < n_features; ++i) {
        local_cssv[i] = local_cssv[i - 1] + local_u[i];
    }

    float c_max = local_cssv[0];
    int rho = 1;
    for (int i = 1; i < n_features; ++i) {
        if (local_u[i] - local_cssv[i] / static_cast<float>(i + 1) > 0.0f) {
            rho = i + 1;
            c_max = local_cssv[i];
        } else {
            break;
        }
    }

    float theta = c_max / static_cast<float>(rho);
    *val = -gamma / 2.0f;
    for (int i = 0; i < n_features; ++i) {
        grads[i] = fmaxf(local_x[i] - theta, 0.0f);
        *val -= grads[i] * (local_x[i] - 0.5f * grads[i]) * gamma;
    }
}

__device__ void smoothmin_device(float* d_dir, int n_steps, float gamma, float* val, float* grads) {
    float soft_val = 0.0f;
    float soft_grad[MAX_STEPS];
    for (int i = 0; i < n_steps; ++i) {
        d_dir[i] = fminf(d_dir[i], 1e20f);
    }
    softmin_device(d_dir, n_steps, gamma, &soft_val, soft_grad);

    *val = 0.0f;
    for (int i = 0; i < n_steps; ++i) {
        *val += d_dir[i] * soft_grad[i];
    }
    for (int i = 0; i < n_steps; ++i) {
        grads[i] = soft_grad[i] * (1.0f - (d_dir[i] - *val) / gamma);
    }
}

__device__ void min_dispatch_device(float* d_dir, int n_steps, float gamma, int min_func, float* val, float* grads) {
    if (min_func == MINFUNC_SOFTMIN) {
        softmin_device(d_dir, n_steps, gamma, val, grads);
    } else if (min_func == MINFUNC_HARDMIN) {
        hardmin_device(d_dir, n_steps, val, grads);
    } else if (min_func == MINFUNC_SMOOTHMIN) {
        smoothmin_device(d_dir, n_steps, gamma, val, grads);
    } else {
        sparsemin_device(d_dir, n_steps, gamma, val, grads);
    }
}

__global__ void mse_cost_forward_kernel(const float* __restrict__ X, const float* __restrict__ Y,
                                        float* __restrict__ C, int total, int N, int M, int D, int Q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int m = idx % M;
    int tmp = idx / M;
    int n = tmp % N;
    int b = tmp / N;
    float cost = 0.0f;
    for (int d = 0; d < D; ++d) {
        for (int q = 0; q < Q; ++q) {
            int xidx = ((b * N + n) * D + d) * Q + q;
            int yidx = ((b * M + m) * D + d) * Q + q;
            float diff = X[xidx] - Y[yidx];
            cost += diff * diff;
        }
    }
    C[idx] = cost;
}

__global__ void bce_cost_forward_kernel(const float* __restrict__ logX, const float* __restrict__ log1minusX,
                                        const float* __restrict__ Y, float* __restrict__ C,
                                        int total, int N, int M, int D, int Q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int m = idx % M;
    int tmp = idx / M;
    int n = tmp % N;
    int b = tmp / N;
    float cost = 0.0f;
    for (int d = 0; d < D; ++d) {
        for (int q = 0; q < Q; ++q) {
            int xidx = ((b * N + n) * D + d) * Q + q;
            int yidx = ((b * M + m) * D + d) * Q + q;
            float y = Y[yidx];
            cost -= y * logX[xidx] + (1.0f - y) * log1minusX[xidx];
        }
    }
    C[idx] = cost;
}

__global__ void ctc_cost_forward_kernel(const float* __restrict__ X, const float* __restrict__ Y,
                                        float* __restrict__ C, int total, int N, int M, int D, int Q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int m = idx % M;
    int tmp = idx / M;
    int n = tmp % N;
    int b = tmp / N;
    float cost = 0.0f;
    for (int d = 0; d < D; ++d) {
        for (int q = 0; q < Q; ++q) {
            int xidx = ((b * N + n) * D + d) * Q + q;
            int yidx = ((b * M + m) * D + d) * Q + q;
            cost -= X[xidx] * Y[yidx];
        }
    }
    C[idx] = cost;
}

__global__ void ctc_step_weights_kernel(const int64_t* __restrict__ targets,
                                        const int64_t* __restrict__ list_M,
                                        const float* __restrict__ global_step_weights,
                                        float blank_penalty,
                                        float* __restrict__ W,
                                        int64_t total,
                                        int N,
                                        int M,
                                        int M_e) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int s = static_cast<int>(idx % 3);
    int64_t tmp = idx / 3;
    int m = static_cast<int>(tmp % M_e);
    tmp /= M_e;
    tmp /= N;
    int b = static_cast<int>(tmp);

    int target_length = static_cast<int>(list_M[b]);
    bool is_blank = (m % 2) == 0;
    float weight = is_blank ? blank_penalty : global_step_weights[s];

    if (s == 2 && m <= 2 * target_length) {
        if (is_blank) {
            weight = 1e20f;
        } else {
            int target_idx = m / 2;
            if (target_idx > 0 && target_idx < target_length &&
                targets[b * M + target_idx] == targets[b * M + target_idx - 1]) {
                weight = 1e20f;
            }
        }
    }

    if (s == 1 && is_blank) {
        int target_idx = m / 2;
        if (target_idx > 0 && target_idx < target_length &&
            targets[b * M + target_idx] == targets[b * M + target_idx - 1]) {
            weight = global_step_weights[1];
        }
    }

    W[idx] = weight;
}

__global__ void ctc_structure_kernel(const int64_t* __restrict__ targets,
                                     const int64_t* __restrict__ list_N,
                                     const int64_t* __restrict__ list_M,
                                     float* __restrict__ Y_e,
                                     int16_t* __restrict__ list_M_e,
                                     int16_t* __restrict__ B_start,
                                     int16_t* __restrict__ B_end,
                                     int64_t total,
                                     int M,
                                     int M_e,
                                     int D,
                                     int blank_index) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int b = static_cast<int>(idx / M_e);
    int m_e = static_cast<int>(idx % M_e);
    int target_length = static_cast<int>(list_M[b]);

    if (m_e <= 2 * target_length) {
        int label = (m_e % 2 == 0)
            ? blank_index
            : static_cast<int>(targets[b * M + m_e / 2]);
        Y_e[(idx * D) + label] = 1.0f;
    }

    if (m_e == 0) {
        int16_t last_n = static_cast<int16_t>(list_N[b] - 1);
        int16_t last_blank = static_cast<int16_t>(2 * target_length);

        list_M_e[b] = static_cast<int16_t>(2 * target_length + 1);

        int boundary_offset = b * 4;
        B_start[boundary_offset] = 0;
        B_start[boundary_offset + 1] = 0;
        B_start[boundary_offset + 2] = 0;
        B_start[boundary_offset + 3] = 1;

        B_end[boundary_offset] = last_n;
        B_end[boundary_offset + 1] = last_blank;
        B_end[boundary_offset + 2] = last_n;
        B_end[boundary_offset + 3] = static_cast<int16_t>(last_blank - 1);
    }
}

__global__ void subseq_initialize_kernel(const int64_t* __restrict__ list_N,
                                         const int64_t* __restrict__ list_M,
                                         const float* __restrict__ global_step_weights,
                                         int16_t* __restrict__ B_start,
                                         int16_t* __restrict__ B_end,
                                         float* __restrict__ start_penalty,
                                         float* __restrict__ end_penalty,
                                         int32_t* __restrict__ num_conditions,
                                         int64_t total,
                                         int max_conditions,
                                         bool sub_X,
                                         bool sub_Y,
                                         bool compensate_subseq) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int b = static_cast<int>(idx / max_conditions);
    int i = static_cast<int>(idx % max_conditions);
    int N = static_cast<int>(list_N[b]);
    int M = static_cast<int>(list_M[b]);
    int x_conditions = sub_X && N > 2 ? N - 2 : 0;
    int y_conditions = sub_Y && M > 2 ? M - 2 : 0;
    int count = 1 + x_conditions + y_conditions;

    if (i == 0) {
        num_conditions[b] = count;
    }
    if (i >= count) return;

    int n_start;
    int m_start;
    int n_end;
    int m_end;
    float start_weight;
    float end_weight;

    if (i == 0) {
        n_start = 0;
        m_start = 0;
        n_end = N - 1;
        m_end = M - 1;
        start_weight = 1.0f;
        end_weight = 0.0f;
    } else if (i <= x_conditions) {
        int n = i;
        n_start = n;
        m_start = 0;
        n_end = n;
        m_end = M - 1;
        start_weight = compensate_subseq ? 1.0f + n * global_step_weights[0] : 1.0f;
        end_weight = compensate_subseq ? (N - 1 - n) * global_step_weights[0] : 0.0f;
    } else {
        int m = i - x_conditions;
        n_start = 0;
        m_start = m;
        n_end = N - 1;
        m_end = m;
        start_weight = compensate_subseq ? 1.0f + m * global_step_weights[1] : 1.0f;
        end_weight = compensate_subseq ? (M - 1 - m) * global_step_weights[1] : 0.0f;
    }

    int64_t boundary_offset = idx * 2;
    B_start[boundary_offset] = static_cast<int16_t>(n_start);
    B_start[boundary_offset + 1] = static_cast<int16_t>(m_start);
    B_end[boundary_offset] = static_cast<int16_t>(n_end);
    B_end[boundary_offset + 1] = static_cast<int16_t>(m_end);
    start_penalty[idx] = start_weight;
    end_penalty[idx] = end_weight;
}

__global__ void mse_grad_x_kernel(const float* __restrict__ X, const float* __restrict__ Y,
                                  const float* __restrict__ grad_C, float* __restrict__ grad_X,
                                  int total, int N, int M, int D, int Q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int q = idx % Q;
    int tmp = idx / Q;
    int d = tmp % D;
    tmp /= D;
    int n = tmp % N;
    int b = tmp / N;
    int xidx = ((b * N + n) * D + d) * Q + q;
    float x = X[xidx];
    float grad = 0.0f;
    for (int m = 0; m < M; ++m) {
        int yidx = ((b * M + m) * D + d) * Q + q;
        grad += 2.0f * (x - Y[yidx]) * grad_C[idx3(b, n, m, N, M)];
    }
    grad_X[xidx] = grad;
}

__global__ void mse_grad_y_kernel(const float* __restrict__ X, const float* __restrict__ Y,
                                  const float* __restrict__ grad_C, float* __restrict__ grad_Y,
                                  int total, int N, int M, int D, int Q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int q = idx % Q;
    int tmp = idx / Q;
    int d = tmp % D;
    tmp /= D;
    int m = tmp % M;
    int b = tmp / M;
    int yidx = ((b * M + m) * D + d) * Q + q;
    float y = Y[yidx];
    float grad = 0.0f;
    for (int n = 0; n < N; ++n) {
        int xidx = ((b * N + n) * D + d) * Q + q;
        grad += 2.0f * (y - X[xidx]) * grad_C[idx3(b, n, m, N, M)];
    }
    grad_Y[yidx] = grad;
}

__global__ void bce_grad_x_kernel(const float* __restrict__ X, const float* __restrict__ Y,
                                  const float* __restrict__ grad_C, float* __restrict__ grad_X,
                                  int total, int N, int M, int D, int Q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int q = idx % Q;
    int tmp = idx / Q;
    int d = tmp % D;
    tmp /= D;
    int n = tmp % N;
    int b = tmp / N;
    int xidx = ((b * N + n) * D + d) * Q + q;
    float x = X[xidx];
    float grad = 0.0f;
    for (int m = 0; m < M; ++m) {
        int yidx = ((b * M + m) * D + d) * Q + q;
        float y = Y[yidx];
        grad += (-(y / x) + (1.0f - y) / (1.0f - x)) * grad_C[idx3(b, n, m, N, M)];
    }
    grad_X[xidx] = grad;
}

__global__ void bce_grad_y_kernel(const float* __restrict__ logX, const float* __restrict__ log1minusX,
                                  const float* __restrict__ grad_C, float* __restrict__ grad_Y,
                                  int total, int N, int M, int D, int Q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int q = idx % Q;
    int tmp = idx / Q;
    int d = tmp % D;
    tmp /= D;
    int m = tmp % M;
    int b = tmp / M;
    int yidx = ((b * M + m) * D + d) * Q + q;
    float grad = 0.0f;
    for (int n = 0; n < N; ++n) {
        int xidx = ((b * N + n) * D + d) * Q + q;
        grad += (-logX[xidx] + log1minusX[xidx]) * grad_C[idx3(b, n, m, N, M)];
    }
    grad_Y[yidx] = grad;
}

__global__ void ctc_grad_x_kernel(const float* __restrict__ Y, const float* __restrict__ grad_C,
                                  float* __restrict__ grad_X, int total, int N, int M, int D, int Q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int q = idx % Q;
    int tmp = idx / Q;
    int d = tmp % D;
    tmp /= D;
    int n = tmp % N;
    int b = tmp / N;
    int xidx = ((b * N + n) * D + d) * Q + q;
    float grad = 0.0f;
    for (int m = 0; m < M; ++m) {
        int yidx = ((b * M + m) * D + d) * Q + q;
        grad -= Y[yidx] * grad_C[idx3(b, n, m, N, M)];
    }
    grad_X[xidx] = grad;
}

__global__ void ctc_grad_y_kernel(const float* __restrict__ X, const float* __restrict__ grad_C,
                                  float* __restrict__ grad_Y, int total, int N, int M, int D, int Q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int q = idx % Q;
    int tmp = idx / Q;
    int d = tmp % D;
    tmp /= D;
    int m = tmp % M;
    int b = tmp / M;
    int yidx = ((b * M + m) * D + d) * Q + q;
    float grad = 0.0f;
    for (int n = 0; n < N; ++n) {
        int xidx = ((b * N + n) * D + d) * Q + q;
        grad -= X[xidx] * grad_C[idx3(b, n, m, N, M)];
    }
    grad_Y[yidx] = grad;
}

__global__ void init_start_kernel(const float* __restrict__ C, float* __restrict__ C_start,
                                  const int16_t* __restrict__ B_start, const float* __restrict__ W_start,
                                  const int16_t* __restrict__ list_N, const int16_t* __restrict__ list_M,
                                  const int32_t* __restrict__ num_start_conditions,
                                  int B, int N, int M, int max_start) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * max_start;
    if (idx >= total) return;
    int i = idx % max_start;
    int b = idx / max_start;
    if (i >= static_cast<int>(num_start_conditions[b])) return;
    int n = static_cast<int>(B_start[(b * max_start + i) * 2]);
    int m = static_cast<int>(B_start[(b * max_start + i) * 2 + 1]);
    if (n < static_cast<int>(list_N[b]) && m < static_cast<int>(list_M[b])) {
        C_start[idx3(b, n, m, N, M)] = C[idx3(b, n, m, N, M)] * W_start[b * max_start + i];
    }
}

__global__ void ddtw_forward_kernel(const float* __restrict__ C, float* __restrict__ D,
                                    float* __restrict__ G, float* __restrict__ GE, float* __restrict__ K,
                                    const float* __restrict__ W, const float* __restrict__ C_start,
                                    const int16_t* __restrict__ B_end, const float* __restrict__ W_end,
                                    float* __restrict__ cost_end, float* __restrict__ grad_end,
                                    float* __restrict__ cost_out, float gamma, int min_func,
                                    const int16_t* __restrict__ list_N, const int16_t* __restrict__ list_M,
                                    const int16_t* __restrict__ step_sizes,
                                    const int32_t* __restrict__ num_end_conditions,
                                    int B, int N, int M, int S, int max_end, int number_of_diagonals) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    float d_dir[MAX_STEPS];
    float grads[MAX_STEPS];
    int n_steps = S + 1;

    for (int p = 0; p < number_of_diagonals; ++p) {
        int n;
        int m;
        if (list_M[b] <= list_N[b]) {
            m = tid;
            n = p - m;
        } else {
            n = tid;
            m = p - n;
        }

        if (!(n < 0 || m < 0 || n >= list_N[b] || m >= list_M[b] || tid > p)) {
            for (int s = 0; s < S; ++s) {
                int n_prev = n - static_cast<int>(step_sizes[s * 2]);
                int m_prev = m - static_cast<int>(step_sizes[s * 2 + 1]);
                if (n_prev >= 0 && m_prev >= 0) {
                    d_dir[s] = W[idx4(b, n, m, s, N, M, S)] * C[idx3(b, n, m, N, M)] + D[idx3(b, n_prev, m_prev, N, M)];
                } else {
                    d_dir[s] = INFINITY;
                }
                d_dir[S] = C_start[idx3(b, n, m, N, M)];
            }

            float val = 0.0f;
            min_dispatch_device(d_dir, n_steps, gamma, min_func, &val, grads);
            D[idx3(b, n, m, N, M)] = val;
            for (int s = 0; s < n_steps; ++s) {
                K[idx4(b, n, m, s, N, M, n_steps)] = grads[s];
            }

            float g = grads[S];
            for (int s = 0; s < S; ++s) {
                g += grads[s] * W[idx4(b, n, m, s, N, M, S)];
            }
            G[idx3(b, n, m, N, M)] = g;
        }
        __syncthreads();
    }

    if (tid == 0) {
        int n_end = static_cast<int>(num_end_conditions[b]);
        for (int i = 0; i < n_end; ++i) {
            int n = static_cast<int>(B_end[(b * max_end + i) * 2]);
            int m = static_cast<int>(B_end[(b * max_end + i) * 2 + 1]);
            cost_end[b * max_end + i] = D[idx3(b, n, m, N, M)] + W_end[b * max_end + i] * C[idx3(b, n, m, N, M)];
        }

        if (max_end <= MAX_STEPS) {
            float val = 0.0f;
            float end_grads[MAX_STEPS];
            min_dispatch_device(&cost_end[b * max_end], max_end, gamma, min_func, &val, end_grads);
            cost_out[b] = val;
            for (int i = 0; i < max_end; ++i) {
                grad_end[b * max_end + i] = end_grads[i];
            }
            for (int i = 0; i < n_end; ++i) {
                int n = static_cast<int>(B_end[(b * max_end + i) * 2]);
                int m = static_cast<int>(B_end[(b * max_end + i) * 2 + 1]);
                GE[idx3(b, n, m, N, M)] = grad_end[b * max_end + i];
            }
        }
    }
}

__global__ void endpoint_min_large_kernel(const float* __restrict__ cost_end,
                                          float* __restrict__ GE,
                                          float* __restrict__ cost_out,
                                          const int16_t* __restrict__ B_end,
                                          const int32_t* __restrict__ num_end_conditions,
                                          float gamma, int min_func,
                                          int N, int M, int max_end) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int n_end = static_cast<int>(num_end_conditions[b]);
    __shared__ float s_val[THREADS_PER_BLOCK];
    __shared__ float s_aux[THREADS_PER_BLOCK];
    __shared__ int s_idx[THREADS_PER_BLOCK];
    __shared__ int s_all_inf[THREADS_PER_BLOCK];
    __shared__ float shared_min;
    __shared__ float shared_sum;
    __shared__ float shared_weighted;
    __shared__ int shared_min_idx;

    if (min_func == MINFUNC_HARDMIN) {
        float local_min = INFINITY;
        int local_idx = 0;
        int local_all_inf = 1;
        for (int i = tid; i < n_end; i += blockDim.x) {
            float v = cost_end[b * max_end + i];
            local_all_inf = local_all_inf && isinf(v);
            if (v < local_min) {
                local_min = v;
                local_idx = i;
            }
        }
        s_val[tid] = local_min;
        s_idx[tid] = local_idx;
        s_all_inf[tid] = local_all_inf;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_all_inf[tid] = s_all_inf[tid] && s_all_inf[tid + stride];
                float other_val = s_val[tid + stride];
                int other_idx = s_idx[tid + stride];
                if (other_val < s_val[tid] || (other_val == s_val[tid] && other_idx < s_idx[tid])) {
                    s_val[tid] = other_val;
                    s_idx[tid] = other_idx;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            shared_min = s_val[0];
            shared_min_idx = s_idx[0];
            cost_out[b] = s_all_inf[0] ? INFINITY : shared_min;
        }
        __syncthreads();

        float all_inf_grad = 1.0f / static_cast<float>(n_end);
        for (int i = tid; i < n_end; i += blockDim.x) {
            int n = static_cast<int>(B_end[(b * max_end + i) * 2]);
            int m = static_cast<int>(B_end[(b * max_end + i) * 2 + 1]);
            float grad = s_all_inf[0] ? all_inf_grad : (i == shared_min_idx ? 1.0f : 0.0f);
            GE[idx3(b, n, m, N, M)] = grad;
        }
        return;
    }

    float local_min = INFINITY;
    int local_all_inf = 1;
    for (int i = tid; i < n_end; i += blockDim.x) {
        float v = cost_end[b * max_end + i];
        if (min_func == MINFUNC_SMOOTHMIN) {
            v = fminf(v, 1e20f);
        }
        local_all_inf = local_all_inf && isinf(v);
        local_min = fminf(local_min, v);
    }
    s_val[tid] = local_min;
    s_all_inf[tid] = local_all_inf;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_all_inf[tid] = s_all_inf[tid] && s_all_inf[tid + stride];
            s_val[tid] = fminf(s_val[tid], s_val[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared_min = s_val[0];
    }
    __syncthreads();

    if (min_func == MINFUNC_SOFTMIN && s_all_inf[0]) {
        if (tid == 0) {
            cost_out[b] = INFINITY;
        }
        __syncthreads();

        float grad = 1.0f / static_cast<float>(n_end);
        for (int i = tid; i < n_end; i += blockDim.x) {
            int n = static_cast<int>(B_end[(b * max_end + i) * 2]);
            int m = static_cast<int>(B_end[(b * max_end + i) * 2 + 1]);
            GE[idx3(b, n, m, N, M)] = grad;
        }
        return;
    }

    float local_sum = 0.0f;
    float local_weighted = 0.0f;
    for (int i = tid; i < n_end; i += blockDim.x) {
        float v = cost_end[b * max_end + i];
        if (min_func == MINFUNC_SMOOTHMIN) {
            v = fminf(v, 1e20f);
        }
        float e = expf(-(v - shared_min) / gamma);
        local_sum += e;
        local_weighted += v * e;
    }
    s_val[tid] = local_sum;
    s_aux[tid] = local_weighted;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_val[tid] += s_val[tid + stride];
            s_aux[tid] += s_aux[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared_sum = s_val[0];
        shared_weighted = s_aux[0] / shared_sum;
        if (min_func == MINFUNC_SOFTMIN) {
            cost_out[b] = -gamma * (-shared_min / gamma + logf(shared_sum));
        } else {
            cost_out[b] = shared_weighted;
        }
    }
    __syncthreads();

    for (int i = tid; i < n_end; i += blockDim.x) {
        int n = static_cast<int>(B_end[(b * max_end + i) * 2]);
        int m = static_cast<int>(B_end[(b * max_end + i) * 2 + 1]);
        float v = cost_end[b * max_end + i];
        if (min_func == MINFUNC_SMOOTHMIN) {
            v = fminf(v, 1e20f);
        }
        float p = expf(-(v - shared_min) / gamma) / shared_sum;
        float grad = p;
        if (min_func == MINFUNC_SMOOTHMIN) {
            grad = p * (1.0f - (v - shared_weighted) / gamma);
        }
        GE[idx3(b, n, m, N, M)] = grad;
    }
}

__global__ void ddtw_backward_kernel(float* __restrict__ E, const float* __restrict__ GE,
                                     const float* __restrict__ G, const float* __restrict__ K,
                                     const float* __restrict__ grad_output, float* __restrict__ grad_C,
                                     const int16_t* __restrict__ list_N, const int16_t* __restrict__ list_M,
                                     const int16_t* __restrict__ step_sizes,
                                     int B, int N, int M, int S, int number_of_diagonals) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int n_steps = S + 1;

    for (int p = 0; p < number_of_diagonals; ++p) {
        int p_rev = number_of_diagonals - p - 1;
        int n;
        int m;
        if (list_M[b] <= list_N[b]) {
            m = tid;
            n = p_rev - m;
        } else {
            n = tid;
            m = p_rev - n;
        }

        if (!(n < 0 || m < 0 || n >= list_N[b] || m >= list_M[b] || tid > p_rev)) {
            float e = 0.0f;
            for (int s = 0; s < S; ++s) {
                int n_next = n + static_cast<int>(step_sizes[s * 2]);
                int m_next = m + static_cast<int>(step_sizes[s * 2 + 1]);
                if (n_next < list_N[b] && m_next < list_M[b]) {
                    e += E[idx3(b, n_next, m_next, N, M)] * K[idx4(b, n_next, m_next, s, N, M, n_steps)];
                }
            }
            e += GE[idx3(b, n, m, N, M)];
            E[idx3(b, n, m, N, M)] = e;
            grad_C[idx3(b, n, m, N, M)] = grad_output[b] * e * G[idx3(b, n, m, N, M)];
        } else if (n >= 0 && m >= 0 && n < N && m < M && !(tid > p_rev)) {
            grad_C[idx3(b, n, m, N, M)] = 0.0f;
        }
        __syncthreads();
    }
}

}  // namespace

torch::Tensor mse_cost_forward_cuda(torch::Tensor X, torch::Tensor Y) {
    CHECK_CUDA(X); CHECK_CUDA(Y); CHECK_FLOAT(X); CHECK_FLOAT(Y);
    X = X.contiguous();
    Y = Y.contiguous();
    int B = X.size(0), N = X.size(1), D = X.size(2), Q = X.size(3), M = Y.size(1);
    auto C = torch::empty({B, N, M}, X.options());
    int total = B * N * M;
    mse_cost_forward_kernel<<<ceil_div(total, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), C.data_ptr<float>(), total, N, M, D, Q);
    return C;
}

torch::Tensor bce_cost_forward_cuda(torch::Tensor logX, torch::Tensor log1minusX, torch::Tensor Y) {
    CHECK_CUDA(logX); CHECK_CUDA(log1minusX); CHECK_CUDA(Y); CHECK_FLOAT(logX); CHECK_FLOAT(log1minusX); CHECK_FLOAT(Y);
    logX = logX.contiguous();
    log1minusX = log1minusX.contiguous();
    Y = Y.contiguous();
    int B = logX.size(0), N = logX.size(1), D = logX.size(2), Q = logX.size(3), M = Y.size(1);
    auto C = torch::empty({B, N, M}, logX.options());
    int total = B * N * M;
    bce_cost_forward_kernel<<<ceil_div(total, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        logX.data_ptr<float>(), log1minusX.data_ptr<float>(), Y.data_ptr<float>(), C.data_ptr<float>(), total, N, M, D, Q);
    return C;
}

torch::Tensor ctc_cost_forward_cuda(torch::Tensor X, torch::Tensor Y) {
    CHECK_CUDA(X); CHECK_CUDA(Y); CHECK_FLOAT(X); CHECK_FLOAT(Y);
    X = X.contiguous();
    Y = Y.contiguous();
    int B = X.size(0), N = X.size(1), D = X.size(2), Q = X.size(3), M = Y.size(1);
    auto C = torch::empty({B, N, M}, X.options());
    int total = B * N * M;
    ctc_cost_forward_kernel<<<ceil_div(total, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), C.data_ptr<float>(), total, N, M, D, Q);
    return C;
}

std::vector<torch::Tensor> ctc_initialize_cuda(torch::Tensor X,
                                               torch::Tensor targets,
                                               torch::Tensor list_N,
                                               torch::Tensor list_M,
                                               torch::Tensor global_step_weights,
                                               double blank_penalty,
                                               int64_t blank_index) {
    CHECK_CUDA(X); CHECK_CUDA(targets); CHECK_CUDA(list_N); CHECK_CUDA(list_M); CHECK_CUDA(global_step_weights);
    CHECK_FLOAT(X); CHECK_FLOAT(global_step_weights);
    TORCH_CHECK(X.dim() == 3, "X must have shape [B, N, D]");
    TORCH_CHECK(targets.dim() == 2, "targets must have shape [B, M]");
    TORCH_CHECK(targets.size(0) == X.size(0), "targets and X must have the same batch size");
    TORCH_CHECK(list_N.dim() == 1 && list_N.size(0) == targets.size(0), "list_N must have shape [B]");
    TORCH_CHECK(list_M.dim() == 1 && list_M.size(0) == targets.size(0), "list_M must have shape [B]");
    TORCH_CHECK(global_step_weights.numel() == 3, "global_step_weights must contain exactly three weights");
    TORCH_CHECK(blank_index >= 0 && blank_index < X.size(2), "blank_index must index a class dimension of X");
    TORCH_CHECK(X.device() == targets.device() && X.device() == list_N.device() &&
                X.device() == list_M.device() && X.device() == global_step_weights.device(),
                "all CTC initialization tensors must be on the same CUDA device");

    targets = targets.to(torch::kInt64).contiguous();
    list_N = list_N.to(torch::kInt64).contiguous();
    list_M = list_M.to(torch::kInt64).contiguous();
    global_step_weights = global_step_weights.contiguous();

    int B = X.size(0);
    int N = X.size(1);
    int D = X.size(2);
    int M = targets.size(1);
    int M_e = 2 * M + 1;
    auto W = torch::empty({B, N, M_e, 3}, X.options());
    auto Y_e = torch::zeros({B, M_e, D}, X.options());
    auto int16_options = targets.options().dtype(torch::kInt16);
    auto list_M_e = torch::empty({B}, int16_options);
    auto B_start = torch::empty({B, 2, 2}, int16_options);
    auto B_end = torch::empty({B, 2, 2}, int16_options);
    int64_t weights_total = W.numel();
    int64_t structure_total = static_cast<int64_t>(B) * M_e;

    ctc_step_weights_kernel<<<ceil_div(weights_total, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        targets.data_ptr<int64_t>(), list_M.data_ptr<int64_t>(), global_step_weights.data_ptr<float>(),
        static_cast<float>(blank_penalty), W.data_ptr<float>(), weights_total, N, M, M_e);
    ctc_structure_kernel<<<ceil_div(structure_total, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        targets.data_ptr<int64_t>(), list_N.data_ptr<int64_t>(), list_M.data_ptr<int64_t>(),
        Y_e.data_ptr<float>(), list_M_e.data_ptr<int16_t>(), B_start.data_ptr<int16_t>(), B_end.data_ptr<int16_t>(),
        structure_total, M, M_e, D, static_cast<int>(blank_index));
    return {W, Y_e, list_M_e, B_start, B_end};
}

std::vector<torch::Tensor> subseq_initialize_cuda(torch::Tensor list_N,
                                                  torch::Tensor list_M,
                                                  torch::Tensor global_step_weights,
                                                  int64_t N_max,
                                                  int64_t M_max,
                                                  bool sub_X,
                                                  bool sub_Y,
                                                  bool compensate_subseq) {
    CHECK_CUDA(list_N); CHECK_CUDA(list_M); CHECK_CUDA(global_step_weights);
    CHECK_FLOAT(global_step_weights);
    TORCH_CHECK(list_N.dim() == 1, "list_N must have shape [B]");
    TORCH_CHECK(list_M.dim() == 1 && list_M.size(0) == list_N.size(0),
                "list_M must have shape [B]");
    TORCH_CHECK(global_step_weights.numel() >= 2, "global_step_weights must contain at least two weights");
    TORCH_CHECK(N_max > 0 && M_max > 0, "maximum sequence lengths must be positive");
    TORCH_CHECK(list_N.device() == list_M.device() && list_N.device() == global_step_weights.device(),
                "all subsequence initialization tensors must be on the same CUDA device");

    list_N = list_N.to(torch::kInt64).contiguous();
    list_M = list_M.to(torch::kInt64).contiguous();
    global_step_weights = global_step_weights.contiguous();

    int B = list_N.size(0);
    int max_x_conditions = sub_X && N_max > 2 ? static_cast<int>(N_max - 2) : 0;
    int max_y_conditions = sub_Y && M_max > 2 ? static_cast<int>(M_max - 2) : 0;
    int max_conditions = 1 + max_x_conditions + max_y_conditions;
    auto int16_options = list_N.options().dtype(torch::kInt16);
    auto B_start = torch::empty({B, max_conditions, 2}, int16_options);
    auto B_end = torch::empty({B, max_conditions, 2}, int16_options);
    auto start_penalty = torch::empty({B, max_conditions}, global_step_weights.options());
    auto end_penalty = torch::empty({B, max_conditions}, global_step_weights.options());
    auto num_conditions = torch::empty({B}, list_N.options().dtype(torch::kInt32));
    int64_t total = static_cast<int64_t>(B) * max_conditions;

    subseq_initialize_kernel<<<ceil_div(total, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        list_N.data_ptr<int64_t>(), list_M.data_ptr<int64_t>(), global_step_weights.data_ptr<float>(),
        B_start.data_ptr<int16_t>(), B_end.data_ptr<int16_t>(), start_penalty.data_ptr<float>(),
        end_penalty.data_ptr<float>(), num_conditions.data_ptr<int32_t>(), total, max_conditions,
        sub_X, sub_Y, compensate_subseq);
    return {B_start, B_end, start_penalty, end_penalty, num_conditions};
}

std::vector<torch::Tensor> mse_cost_backward_cuda(torch::Tensor X, torch::Tensor Y, torch::Tensor grad_C) {
    CHECK_CUDA(X); CHECK_CUDA(Y); CHECK_CUDA(grad_C); CHECK_FLOAT(X); CHECK_FLOAT(Y); CHECK_FLOAT(grad_C);
    X = X.contiguous();
    Y = Y.contiguous();
    grad_C = grad_C.contiguous();
    int B = X.size(0), N = X.size(1), D = X.size(2), Q = X.size(3), M = Y.size(1);
    auto grad_X = torch::empty_like(X);
    auto grad_Y = torch::empty_like(Y);
    int total_x = B * N * D * Q;
    int total_y = B * M * D * Q;
    mse_grad_x_kernel<<<ceil_div(total_x, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), grad_C.data_ptr<float>(), grad_X.data_ptr<float>(), total_x, N, M, D, Q);
    mse_grad_y_kernel<<<ceil_div(total_y, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), grad_C.data_ptr<float>(), grad_Y.data_ptr<float>(), total_y, N, M, D, Q);
    return {grad_X, grad_Y};
}

std::vector<torch::Tensor> bce_cost_backward_cuda(torch::Tensor X, torch::Tensor logX, torch::Tensor log1minusX,
                                                  torch::Tensor Y, torch::Tensor grad_C) {
    CHECK_CUDA(X); CHECK_CUDA(logX); CHECK_CUDA(log1minusX); CHECK_CUDA(Y); CHECK_CUDA(grad_C);
    CHECK_FLOAT(X); CHECK_FLOAT(logX); CHECK_FLOAT(log1minusX); CHECK_FLOAT(Y); CHECK_FLOAT(grad_C);
    X = X.contiguous();
    logX = logX.contiguous();
    log1minusX = log1minusX.contiguous();
    Y = Y.contiguous();
    grad_C = grad_C.contiguous();
    int B = X.size(0), N = X.size(1), D = X.size(2), Q = X.size(3), M = Y.size(1);
    auto grad_X = torch::empty_like(X);
    auto grad_Y = torch::empty_like(Y);
    int total_x = B * N * D * Q;
    int total_y = B * M * D * Q;
    bce_grad_x_kernel<<<ceil_div(total_x, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), grad_C.data_ptr<float>(), grad_X.data_ptr<float>(), total_x, N, M, D, Q);
    bce_grad_y_kernel<<<ceil_div(total_y, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        logX.data_ptr<float>(), log1minusX.data_ptr<float>(), grad_C.data_ptr<float>(), grad_Y.data_ptr<float>(), total_y, N, M, D, Q);
    return {grad_X, grad_Y};
}

std::vector<torch::Tensor> ctc_cost_backward_cuda(torch::Tensor X, torch::Tensor Y, torch::Tensor grad_C) {
    CHECK_CUDA(X); CHECK_CUDA(Y); CHECK_CUDA(grad_C); CHECK_FLOAT(X); CHECK_FLOAT(Y); CHECK_FLOAT(grad_C);
    X = X.contiguous();
    Y = Y.contiguous();
    grad_C = grad_C.contiguous();
    int B = X.size(0), N = X.size(1), D = X.size(2), Q = X.size(3), M = Y.size(1);
    auto grad_X = torch::empty_like(X);
    auto grad_Y = torch::empty_like(Y);
    int total_x = B * N * D * Q;
    int total_y = B * M * D * Q;
    ctc_grad_x_kernel<<<ceil_div(total_x, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        Y.data_ptr<float>(), grad_C.data_ptr<float>(), grad_X.data_ptr<float>(), total_x, N, M, D, Q);
    ctc_grad_y_kernel<<<ceil_div(total_y, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), grad_C.data_ptr<float>(), grad_Y.data_ptr<float>(), total_y, N, M, D, Q);
    return {grad_X, grad_Y};
}

std::vector<torch::Tensor> ddtw_forward_cuda(torch::Tensor C,
                                             int64_t min_func,
                                             double gamma,
                                             torch::Tensor step_sizes,
                                             torch::Tensor W,
                                             torch::Tensor list_N,
                                             torch::Tensor list_M,
                                             torch::Tensor B_start,
                                             torch::Tensor B_end,
                                             torch::Tensor num_start_conditions,
                                             torch::Tensor num_end_conditions,
                                             torch::Tensor W_start,
                                             torch::Tensor W_end) {
    CHECK_CUDA(C); CHECK_CUDA(step_sizes); CHECK_CUDA(W); CHECK_CUDA(list_N); CHECK_CUDA(list_M);
    CHECK_CUDA(B_start); CHECK_CUDA(B_end); CHECK_CUDA(num_start_conditions); CHECK_CUDA(num_end_conditions); CHECK_CUDA(W_start); CHECK_CUDA(W_end);
    CHECK_FLOAT(C); CHECK_FLOAT(W); CHECK_FLOAT(W_start); CHECK_FLOAT(W_end);
    CHECK_INT32(num_start_conditions); CHECK_INT32(num_end_conditions);
    C = C.contiguous();
    W = W.contiguous();
    step_sizes = step_sizes.contiguous();
    list_N = list_N.contiguous();
    list_M = list_M.contiguous();
    B_start = B_start.contiguous();
    B_end = B_end.contiguous();
    num_start_conditions = num_start_conditions.contiguous();
    num_end_conditions = num_end_conditions.contiguous();
    W_start = W_start.contiguous();
    W_end = W_end.contiguous();

    int B = C.size(0), N = C.size(1), M = C.size(2), S = step_sizes.size(0);
    int max_start = B_start.size(1);
    int max_end = B_end.size(1);
    int elements_per_diagonal = std::min(N, M);
    int number_of_diagonals = N + M - 1;

    TORCH_CHECK(elements_per_diagonal <= 1024, "min(N, M) must be <= 1024");
    TORCH_CHECK(S + 1 <= MAX_STEPS, "number of steps + 1 exceeds MAX_STEPS");
    TORCH_CHECK(max_end <= MAX_STEPS || min_func != MINFUNC_SPARSEMIN,
                "sparsemin with more than MAX_STEPS end conditions is not supported");

    auto C_start = torch::full_like(C, std::numeric_limits<float>::infinity());
    auto GE = torch::zeros_like(C);
    auto cost_end = torch::full({B, max_end}, std::numeric_limits<float>::infinity(), C.options());
    auto grad_end = torch::zeros({B, max_end}, C.options());
    auto D = torch::zeros_like(C);
    auto G = torch::zeros_like(C);
    auto K = torch::zeros({B, N, M, S + 1}, C.options());
    auto cost_out = torch::zeros({B}, C.options());

    int start_total = B * max_start;
    init_start_kernel<<<ceil_div(start_total, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        C.data_ptr<float>(), C_start.data_ptr<float>(), B_start.data_ptr<int16_t>(), W_start.data_ptr<float>(),
        list_N.data_ptr<int16_t>(), list_M.data_ptr<int16_t>(), num_start_conditions.data_ptr<int32_t>(), B, N, M, max_start);

    ddtw_forward_kernel<<<B, elements_per_diagonal, 0, at::cuda::getCurrentCUDAStream()>>>(
        C.data_ptr<float>(), D.data_ptr<float>(), G.data_ptr<float>(), GE.data_ptr<float>(), K.data_ptr<float>(),
        W.data_ptr<float>(), C_start.data_ptr<float>(), B_end.data_ptr<int16_t>(), W_end.data_ptr<float>(),
        cost_end.data_ptr<float>(), grad_end.data_ptr<float>(), cost_out.data_ptr<float>(),
        static_cast<float>(gamma), static_cast<int>(min_func), list_N.data_ptr<int16_t>(), list_M.data_ptr<int16_t>(),
        step_sizes.data_ptr<int16_t>(), num_end_conditions.data_ptr<int32_t>(), B, N, M, S, max_end, number_of_diagonals);

    if (max_end > MAX_STEPS) {
        endpoint_min_large_kernel<<<B, THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            cost_end.data_ptr<float>(), GE.data_ptr<float>(), cost_out.data_ptr<float>(),
            B_end.data_ptr<int16_t>(), num_end_conditions.data_ptr<int32_t>(),
            static_cast<float>(gamma), static_cast<int>(min_func), N, M, max_end);
    }

    return {cost_out, G, K, GE, D, C_start};
}

std::vector<torch::Tensor> ddtw_backward_cuda(torch::Tensor G,
                                              torch::Tensor K,
                                              torch::Tensor GE,
                                              torch::Tensor grad_output,
                                              torch::Tensor list_N,
                                              torch::Tensor list_M,
                                              torch::Tensor step_sizes) {
    CHECK_CUDA(G); CHECK_CUDA(K); CHECK_CUDA(GE); CHECK_CUDA(grad_output); CHECK_CUDA(list_N); CHECK_CUDA(list_M); CHECK_CUDA(step_sizes);
    CHECK_FLOAT(G); CHECK_FLOAT(K); CHECK_FLOAT(GE); CHECK_FLOAT(grad_output);
    G = G.contiguous();
    K = K.contiguous();
    GE = GE.contiguous();
    grad_output = grad_output.contiguous();
    list_N = list_N.contiguous();
    list_M = list_M.contiguous();
    step_sizes = step_sizes.contiguous();

    int B = G.size(0), N = G.size(1), M = G.size(2), S = step_sizes.size(0);
    int elements_per_diagonal = std::min(N, M);
    int number_of_diagonals = N + M - 1;
    auto E = torch::zeros_like(G);
    auto grad_C = torch::zeros_like(G);

    ddtw_backward_kernel<<<B, elements_per_diagonal, 0, at::cuda::getCurrentCUDAStream()>>>(
        E.data_ptr<float>(), GE.data_ptr<float>(), G.data_ptr<float>(), K.data_ptr<float>(),
        grad_output.data_ptr<float>(), grad_C.data_ptr<float>(), list_N.data_ptr<int16_t>(), list_M.data_ptr<int16_t>(),
        step_sizes.data_ptr<int16_t>(), B, N, M, S, number_of_diagonals);

    return {grad_C, E};
}
