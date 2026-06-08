#include <iostream>
#include <vector>
#include <thread>
#include <immintrin.h> // For SIMD intrinsics
#include <omp.h>       // For OpenMP parallelization

// Apply dilation on a single 2D image
void dilateSingleImage(std::vector<std::vector<int>>& image, int kernelSize) {
    int rows = image.size();
    int cols = image[0].size();
    int pad = kernelSize / 2;

    std::vector<std::vector<int>> result(rows, std::vector<int>(cols, 0));

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int maxVal = 0;

            // SIMD-friendly loop over kernel
            for (int ki = -pad; ki <= pad; ki++) {
                for (int kj = -pad; kj <= pad; kj++) {
                    int ni = i + ki;
                    int nj = j + kj;
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                        maxVal = std::max(maxVal, image[ni][nj]);
                    }
                }
            }
            result[i][j] = maxVal;
        }
    }

    image = result;
}

// Apply dilation on a collection of images
void morphologicalDilation(std::vector<std::vector<std::vector<int>>>& images, int kernelSize) {
    #pragma omp parallel for
    for (size_t idx = 0; idx < images.size(); idx++) {
        dilateSingleImage(images[idx], kernelSize);
    }
}

// Simple test harness
int main() {
    // Example: two binary images
    std::vector<std::vector<std::vector<int>>> images = {
        {
            {0,0,1,0,0},
            {0,1,1,1,0},
            {0,0,1,0,0},
            {0,0,0,0,0},
            {0,0,0,0,0}
        },
        {
            {1,0,0,0,1},
            {0,0,0,0,0},
            {0,1,1,1,0},
            {0,0,0,0,0},
            {1,0,0,0,1}
        }
    };

    int kernelSize = 3;
    morphologicalDilation(images, kernelSize);

    // Print results
    for (size_t idx = 0; idx < images.size(); idx++) {
        std::cout << "Image " << idx << " after dilation:\n";
        for (auto& row : images[idx]) {
            for (auto val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}
