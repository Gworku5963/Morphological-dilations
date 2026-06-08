#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>

// Perform morphological dilation on a collection of binary images
// Each image is a 2D vector<int> with values 0 or 1
// Kernel size must be odd (3, 5, 7, ...)
std::vector<std::vector<std::vector<int>>> morphologicalDilation(
    const std::vector<std::vector<std::vector<int>>>& images, int kernelSize) {

    int pad = kernelSize / 2;
    std::vector<std::vector<std::vector<int>>> result(images.size());

    #pragma omp parallel for
    for (size_t idx = 0; idx < images.size(); ++idx) {
        const auto& img = images[idx];
        int rows = img.size();
        int cols = img[0].size();
        result[idx].assign(rows, std::vector<int>(cols, 0));

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int maxVal = 0;
                for (int ki = -pad; ki <= pad; ++ki) {
                    for (int kj = -pad; kj <= pad; ++kj) {
                        int ni = i + ki;
                        int nj = j + kj;
                        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                            maxVal = std::max(maxVal, img[ni][nj]);
                        }
                    }
                }
                result[idx][i][j] = maxVal;
            }
        }
    }
    return result;
}
int main() {
    // Example: two small binary images
    std::vector<std::vector<std::vector<int>>> images = {
        { {0,0,0,0,0},
          {0,1,0,0,0},
          {0,0,1,0,0},
          {0,0,0,0,0},
          {0,0,0,0,0} },

        { {0,0,0},
          {0,1,0},
          {0,0,0} }
    };

    int kernelSize = 3;
    auto dilated = morphologicalDilation(images, kernelSize);

    // Print results
    for (size_t idx = 0; idx < dilated.size(); ++idx) {
        std::cout << "Image " << idx << " after dilation:\n";
        for (auto& row : dilated[idx]) {
            for (auto val : row) std::cout << val << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    return 0;
}
