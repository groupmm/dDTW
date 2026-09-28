#include <torch/extension.h>

#include <vector>

torch::Tensor mse_cost_forward_cuda(torch::Tensor X, torch::Tensor Y);
torch::Tensor bce_cost_forward_cuda(torch::Tensor logX, torch::Tensor log1minusX, torch::Tensor Y);
torch::Tensor ctc_cost_forward_cuda(torch::Tensor X, torch::Tensor Y);
std::vector<torch::Tensor> ctc_initialize_cuda(torch::Tensor X,
                                               torch::Tensor targets,
                                               torch::Tensor list_N,
                                               torch::Tensor list_M,
                                               torch::Tensor global_step_weights,
                                               double blank_penalty,
                                               int64_t blank_index);
std::vector<torch::Tensor> subseq_initialize_cuda(torch::Tensor list_N,
                                                  torch::Tensor list_M,
                                                  torch::Tensor global_step_weights,
                                                  int64_t N_max,
                                                  int64_t M_max,
                                                  bool sub_X,
                                                  bool sub_Y,
                                                  bool compensate_subseq);

std::vector<torch::Tensor> mse_cost_backward_cuda(torch::Tensor X, torch::Tensor Y, torch::Tensor grad_C);
std::vector<torch::Tensor> bce_cost_backward_cuda(torch::Tensor X, torch::Tensor logX, torch::Tensor log1minusX,
                                                  torch::Tensor Y, torch::Tensor grad_C);
std::vector<torch::Tensor> ctc_cost_backward_cuda(torch::Tensor X, torch::Tensor Y, torch::Tensor grad_C);

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
                                             torch::Tensor W_end);

std::vector<torch::Tensor> ddtw_backward_cuda(torch::Tensor G,
                                              torch::Tensor K,
                                              torch::Tensor GE,
                                              torch::Tensor grad_output,
                                              torch::Tensor list_N,
                                              torch::Tensor list_M,
                                              torch::Tensor step_sizes);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mse_cost_forward", &mse_cost_forward_cuda, "MSE local-cost forward (CUDA)");
    m.def("bce_cost_forward", &bce_cost_forward_cuda, "BCE local-cost forward (CUDA)");
    m.def("ctc_cost_forward", &ctc_cost_forward_cuda, "CTC local-cost forward (CUDA)");
    m.def("ctc_initialize", &ctc_initialize_cuda, "CTC input initialization (CUDA)");
    m.def("subseq_initialize", &subseq_initialize_cuda, "Subsequence boundary initialization (CUDA)");

    m.def("mse_cost_backward", &mse_cost_backward_cuda, "MSE local-cost backward (CUDA)");
    m.def("bce_cost_backward", &bce_cost_backward_cuda, "BCE local-cost backward (CUDA)");
    m.def("ctc_cost_backward", &ctc_cost_backward_cuda, "CTC local-cost backward (CUDA)");

    m.def("ddtw_forward", &ddtw_forward_cuda, "dDTW forward (CUDA)");
    m.def("ddtw_backward", &ddtw_backward_cuda, "dDTW backward (CUDA)");
}
