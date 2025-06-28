
#include <torch/extension.h>

#include <iostream>

torch::Tensor disc_cumsum(const torch::Tensor& input, double gamma) {

    // Input is of size (n_trajectories, n_steps)
    // For each point in a trajectory, we need to compute the discounted sum of the future rewards.
    // Output should be of size (n_trajectories, steps)
    torch::Tensor output = torch::zeros_like(input);
    int num_dims = input.dim();

    if (num_dims == 2) {
        // For multiple trajectories batched in a single torch::Tensor
        int num_trajectories = input.size(0);
        int max_index = input.size(1) - 1;
        for (int i = 0; i < num_trajectories; ++i) {
            std::cout << "Computing trajectory " << i << std::endl;
            for (int j = max_index; j >= 0; --j) {
                if (j == max_index) {
                    output[i][j] = input[i][j];
                } else {
                    output[i][j] = input[i][j] + gamma * output[i][j + 1];
                }
            }
        }
    }
    else {
        // For a single trajectory
        int max_index = input.size(0) - 1;
        for (int j = max_index; j >= 0; --j) {
            if (j == max_index) {
                output[j] = input[j];
            } else {
                output[j] = input[j] + gamma * output[j + 1];
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("disc_cumsum", &disc_cumsum, "disc_cumsum");
}
