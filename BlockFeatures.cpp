#include <opencv2/opencv.hpp>
#include <array>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

#include "BlockFeatures.h"

// =======================================================
// 1D Fast Walsh-Hadamard Transform (in-place)
// =======================================================
inline void fwht_1d(cv::Mat& vec) {
    int n = vec.cols > 1 ? vec.cols : vec.rows;
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = vec.at<float>(i + j);
                float v = vec.at<float>(i + j + len);
                vec.at<float>(i + j) = u + v;
                vec.at<float>(i + j + len) = u - v;
            }
        }
    }
}

// =======================================================
// 2D Hadamard Transform
// =======================================================
inline cv::Mat fwht_2d(const cv::Mat& blk) {
    cv::Mat H;
    blk.convertTo(H, CV_32F);
    // rows
    for (int r = 0; r < H.rows; r++) {
        cv::Mat row = H.row(r);
        fwht_1d(row);
    }
    // columns
    for (int c = 0; c < H.cols; c++) {
        cv::Mat col = H.col(c);
        fwht_1d(col);
    }
    return H;
}

// =======================================================
// 1. FEATURE 1 — Mean, Variance, StdDev and Sum
// =======================================================
inline std::tuple<double,double,double,double> calculate_basic_features_cv(const cv::Mat& blk)
{
    cv::Mat blk_f;
    blk.convertTo(blk_f, CV_32F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(blk_f, mean, stddev);
    double var = stddev[0]*stddev[0];
    double sum = cv::sum(blk_f)[0];
    return {mean[0], var, stddev[0], sum};
}

// =======================================================
// 2. FEATURE 2 — vH, vV, dH, dV
// =======================================================
inline std::array<double,4> calculate_stats_cv(const cv::Mat& blk)
{
    cv::Mat blk_f;
    blk.convertTo(blk_f, CV_32F);

    cv::Mat row_means; cv::reduce(blk_f, row_means, 1, cv::REDUCE_AVG);
    cv::Mat row_means_exp; cv::repeat(row_means, 1, blk_f.cols, row_means_exp);
    cv::Mat diff_row = blk_f - row_means_exp;
    cv::Mat row_vars; cv::reduce(diff_row.mul(diff_row), row_vars, 1, cv::REDUCE_AVG);
    cv::Mat row_stds; cv::sqrt(row_vars, row_stds);
    double vH = cv::mean(row_vars)[0], dH = cv::mean(row_stds)[0];

    cv::Mat col_means; cv::reduce(blk_f, col_means, 0, cv::REDUCE_AVG);
    cv::Mat col_means_exp; cv::repeat(col_means, blk_f.rows, 1, col_means_exp);
    cv::Mat diff_col = blk_f - col_means_exp;
    cv::Mat col_vars; cv::reduce(diff_col.mul(diff_col), col_vars, 0, cv::REDUCE_AVG);
    cv::Mat col_stds; cv::sqrt(col_vars, col_stds);
    double vV = cv::mean(col_vars)[0], dV = cv::mean(col_stds)[0];

    return {vH, vV, dV, dH};
}

// =======================================================
// 3. FEATURE 3 — Sobel Gradients
// =======================================================
inline std::array<double,5> calculate_gradients_sobel_cv(const cv::Mat& blk)
{
    cv::Mat blk_f; blk.convertTo(blk_f, CV_32F);
    cv::Mat Gh, Gv;
    cv::Sobel(blk_f, Gh, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
    cv::Sobel(blk_f, Gv, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
    double mGv = cv::mean(cv::abs(Gv))[0];
    double mGh = cv::mean(cv::abs(Gh))[0];
    cv::Mat Mag, Dir; cv::magnitude(Gv, Gh, Mag); cv::phase(Gh, Gv, Dir, true);
    double meanMag = cv::mean(Mag)[0];
    double meanDir = cv::mean(Dir)[0];
    double razao_grad = mGh/(mGv + 1e-6);
    return {mGv, mGh, meanMag, meanDir, razao_grad};
}

// =======================================================
// 4. FEATURE 4 — Prewitt Gradients
// =======================================================
inline std::array<double,5> calculate_gradients_prewitt_cv(const cv::Mat& blk)
{
    cv::Mat blk_f; blk.convertTo(blk_f, CV_32F);
    cv::Mat kernel_gx = (cv::Mat_<float>(3,3) << -1,0,1,-1,0,1,-1,0,1);
    cv::Mat kernel_gy = (cv::Mat_<float>(3,3) << -1,-1,-1,0,0,0,1,1,1);
    cv::Mat Gh, Gv;
    cv::filter2D(blk_f, Gh, CV_32F, kernel_gx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(blk_f, Gv, CV_32F, kernel_gy, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    double mGv = cv::mean(cv::abs(Gv))[0];
    double mGh = cv::mean(cv::abs(Gh))[0];
    cv::Mat Mag, Dir; cv::magnitude(Gv, Gh, Mag); cv::phase(Gh, Gv, Dir, true);
    double meanMag = cv::mean(Mag)[0];
    double meanDir = cv::mean(Dir)[0];
    double razao_grad = mGh/(mGv + 1e-6);
    return {mGv, mGh, meanMag, meanDir, razao_grad};
}

// =======================================================
// 5. FEATURE 5 — Contrast
// =======================================================
inline std::array<double,3> calculate_contrast_features_cv(const cv::Mat& blk)
{
    double minVal, maxVal;
    cv::minMaxLoc(blk, &minVal, &maxVal);
    return {minVal, maxVal, maxVal - minVal};
}

// =======================================================
// 6. FEATURE 6 — Sharpness (Laplacian variance)
// =======================================================
inline double calculate_laplacian_var_cv(const cv::Mat& blk)
{
    cv::Mat blk_f; 
    blk.convertTo(blk_f, CV_32F);
    cv::Mat lap;
    cv::Laplacian(blk_f, lap, CV_32F, 1, 1, 0);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);     
    return stddev[0]*stddev[0];
}


// =======================================================
// 7. FEATURE 7 — Shannon Entropy
// =======================================================
inline double calculate_entropy_cv(const cv::Mat& blk)
{
    cv::Mat blk_f;
    blk.convertTo(blk_f, CV_32F); // Converte para float para evitar erro no calcHist com CV_16S

    int histSize = 256;
    // VTM usa 10-bit (0-1023). Ajustamos o range para cobrir todos os valores possíveis.
    // O histograma irá agrupar esses valores em 256 bins.
    float range[] = {0, 1024}; 
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&blk_f, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    double sum = cv::sum(hist)[0];
    if (sum > 0) hist /= sum;

    double entropy = 0.0;
    for(int i=0;i<histSize;i++) {
        float p = hist.at<float>(i);
        if(p>0) entropy -= p*log2(p);
    }
    return entropy;
}

// =======================================================
// 8. FEATURE 8 — Hadamard
// =======================================================
inline HadamardFeatures calculate_hadamard_features(const cv::Mat& blk)
{
    cv::Mat H = fwht_2d(blk);
    HadamardFeatures f{};
    f.dc = H.at<float>(0,0);
    f.energy_total = cv::sum(H.mul(H))[0];
    f.energy_ac = f.energy_total - f.dc*f.dc;
    double minVal,maxVal; cv::minMaxLoc(H,&minVal,&maxVal);
    f.min_coef = minVal; f.max_coef = maxVal;
    f.top_left = H.at<float>(0,0);
    f.top_right = H.at<float>(0,H.cols-1);
    f.bottom_left = H.at<float>(H.rows-1,0);
    f.bottom_right = H.at<float>(H.rows-1,H.cols-1);
    return f;
}

// =======================================================
// 9. FEATURE 9 — Residual Features
// =======================================================

inline ResidualFeatures calculate_residual_features(const cv::Mat& resi)
{
    ResidualFeatures f{};
    cv::Mat resi_f;
    resi.convertTo(resi_f, CV_32F);

    // SAD - Soma absoluta dos valores residuais
    f.sad = cv::sum(cv::abs(resi_f))[0];

    // Soma da última linha e última coluna
    cv::Mat last_row = resi_f.row(resi_f.rows - 1);
    cv::Mat last_col = resi_f.col(resi_f.cols - 1);
    f.last_row_sum = cv::sum(last_row)[0];
    f.last_col_sum = cv::sum(last_col)[0];

    // TopLeft, TopRight, BottomRight
    f.top_left     = resi_f.at<float>(0, 0);
    f.top_right    = resi_f.at<float>(0, resi_f.cols - 1);
    f.bottom_right = resi_f.at<float>(resi_f.rows - 1, resi_f.cols - 1);

    return f;
}

// =======================================================
// MAIN STRUCT
// =======================================================
// =======================================================
// MAIN EXTRACTION
// =======================================================
BlockFeatures extract_block_features(const cv::Mat& blk, const cv::Mat& resi)
{
    BlockFeatures f{};
    auto [mean,var,std_dev,sum_val] = calculate_basic_features_cv(blk);
    f.blk_pixel_mean = mean; f.blk_pixel_variance = var; f.blk_pixel_std_dev = std_dev; f.blk_pixel_sum = sum_val;

    auto stats = calculate_stats_cv(blk);
    f.blk_var_h = stats[0]; f.blk_var_v = stats[1]; f.blk_std_v = stats[2]; f.blk_std_h = stats[3];

    auto sob = calculate_gradients_sobel_cv(blk);
    f.blk_sobel_gv = sob[0]; f.blk_sobel_gh = sob[1]; f.blk_sobel_mag = sob[2]; f.blk_sobel_dir = sob[3]; f.blk_sobel_razao_grad = sob[4];

    auto pre = calculate_gradients_prewitt_cv(blk);
    f.blk_prewitt_gv = pre[0]; f.blk_prewitt_gh = pre[1]; f.blk_prewitt_mag = pre[2]; f.blk_prewitt_dir = pre[3]; f.blk_prewitt_razao_grad = pre[4];

    auto contrast = calculate_contrast_features_cv(blk);
    f.blk_min = contrast[0]; f.blk_max = contrast[1]; f.blk_range = contrast[2];

    f.blk_laplacian_var = calculate_laplacian_var_cv(blk);
    f.blk_entropy = calculate_entropy_cv(blk);

    f.hadamard = calculate_hadamard_features(blk);

    // Extração das novas features de resíduo
    f.residual = calculate_residual_features(resi);
    return f;
}

// =======================================================
// PRINT
// =======================================================
void print_features(const BlockFeatures& f)
{
    std::cout << std::fixed << std::setprecision(6);
    std::cout << " mean = " << f.blk_pixel_mean << "\n";
    std::cout << " var  = " << f.blk_pixel_variance << "\n";
    std::cout << " std  = " << f.blk_pixel_std_dev << "\n";
    std::cout << " sum  = " << f.blk_pixel_sum << "\n";
    std::cout << " vH   = " << f.blk_var_h << "\n";
    std::cout << " vV   = " << f.blk_var_v << "\n";
    std::cout << " dV   = " << f.blk_std_v << "\n";
    std::cout << " dH   = " << f.blk_std_h << "\n";
    std::cout << " sob_Gv  = " << f.blk_sobel_gv << "\n";
    std::cout << " sob_Gh  = " << f.blk_sobel_gh << "\n";
    std::cout << " sob_mag = " << f.blk_sobel_mag << "\n";
    std::cout << " sob_dir = " << f.blk_sobel_dir << "\n";
    std::cout << " sob_razao = " << f.blk_sobel_razao_grad << "\n";
    std::cout << " pre_Gv  = " << f.blk_prewitt_gv << "\n";
    std::cout << " pre_Gh  = " << f.blk_prewitt_gh << "\n";
    std::cout << " pre_mag = " << f.blk_prewitt_mag << "\n";
    std::cout << " pre_dir = " << f.blk_prewitt_dir << "\n";
    std::cout << " pre_razao = " << f.blk_prewitt_razao_grad << "\n";
    std::cout << " min = " << f.blk_min << ", max = " << f.blk_max << ", range = " << f.blk_range << "\n";
    std::cout << " lap_var = " << f.blk_laplacian_var << "\n";
    std::cout << " entropy = " << f.blk_entropy << "\n";

    std::cout << " H_dc           = " << f.hadamard.dc << "\n";
    std::cout << " H_energy_total = " << f.hadamard.energy_total << "\n";
    std::cout << " H_energy_ac    = " << f.hadamard.energy_ac << "\n";
    std::cout << " H_max          = " << f.hadamard.max_coef << "\n";
    std::cout << " H_min          = " << f.hadamard.min_coef << "\n";
    std::cout << " H_top_left     = " << f.hadamard.top_left << "\n";
    std::cout << " H_top_right    = " << f.hadamard.top_right << "\n";
    std::cout << " H_bottom_left  = " << f.hadamard.bottom_left << "\n";
    std::cout << " H_bottom_right = " << f.hadamard.bottom_right << "\n";
    // Residual
    std::cout << " --- Residual Features ---\n";
    std::cout << " resi_sad      = " << f.residual.sad << "\n";
    std::cout << " resi_last_row = " << f.residual.last_row_sum << "\n";
    std::cout << " resi_last_col = " << f.residual.last_col_sum << "\n";
    std::cout << " resi_TL       = " << f.residual.top_left << "\n";
    std::cout << " resi_TR       = " << f.residual.top_right << "\n";
    std::cout << " resi_BR       = " << f.residual.bottom_right << "\n";
}