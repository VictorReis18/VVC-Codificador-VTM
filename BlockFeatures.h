#ifndef __BLOCK_FEATURES_H__
#define __BLOCK_FEATURES_H__

#include <opencv2/opencv.hpp>
#include <vector>

struct HadamardFeatures {
    double dc;
    double energy_total;
    double energy_ac;
    double max_coef;
    double min_coef;
    double top_left;
    double top_right;
    double bottom_left;
    double bottom_right;
};

struct ResidualFeatures {
    double sad;
    double last_row_sum;
    double last_col_sum;
    double top_left;
    double top_right;
    double bottom_right;
};

struct BlockFeatures {
    double blk_pixel_mean;
    double blk_pixel_variance;
    double blk_pixel_std_dev;
    double blk_pixel_sum;
    double blk_var_h;
    double blk_var_v;
    double blk_std_v;
    double blk_std_h;
    double blk_sobel_gv;
    double blk_sobel_gh;
    double blk_sobel_mag;
    double blk_sobel_dir;
    double blk_sobel_razao_grad;
    double blk_prewitt_gv;
    double blk_prewitt_gh;
    double blk_prewitt_mag;
    double blk_prewitt_dir;
    double blk_prewitt_razao_grad;
    double blk_min;
    double blk_max;
    double blk_range;
    double blk_laplacian_var;
    double blk_entropy;

    // Hadamard
    HadamardFeatures hadamard;

    //Residuos
    ResidualFeatures residual;
};

BlockFeatures extract_block_features(const cv::Mat& blk, const cv::Mat& resi);
void print_features(const BlockFeatures& f);

#endif // __BLOCK_FEATURES_H__