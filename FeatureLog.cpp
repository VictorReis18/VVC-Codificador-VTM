#include "FeatureLog.h"
#include "CommonLib/CodingStructure.h"
#include "CommonLib/Slice.h"
#include "CommonLib/Unit.h"
#include <map>
#include <mutex>
#include <sstream>
#include <locale>

namespace CAROL {

// Estrutura global para evitar conflitos entre threads 
static std::map<std::string, std::string> g_lineBuffer;
static std::mutex g_logMutex;

void FeatureLogger::init(const std::string& inputName, int qp) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    if (m_initialized) return;

    std::string fileName = inputName + "_" + std::to_string(qp) + ".csv";
    m_csvFile.open(fileName, std::ios::app);

    m_csvFile.seekp(0, std::ios::end);

    if (m_csvFile.tellp() == 0) {
        m_csvFile << "POC,X,Y,W,H,QP,"
                  << "Mean,Var,StdDev,Sum,VarH,VarV,StdV,StdH,"
                  << "SobelGV,SobelGH,SobelMag,SobelDir,SobelRatio,"
                  << "PrewittGV,PrewittGH,PrewittMag,PrewittDir,PrewittRatio,"
                  << "Min,Max,Range,LaplacianVar,Entropy,"
                  << "H_DC,H_EnergyTotal,H_EnergyAC,H_Max,H_Min,"
                  << "H_TL,H_TR,H_BL,H_BR,"
                  << "SizeGroup,Area,Orientation,AspectRatioIdx,"
                  << "Transformada" << std::endl;
    }
    m_initialized = true;
}

void FeatureLogger::startLine(const PredictionUnit& pu, const BlockFeatures& feats, int baseQP) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    
    if (!m_csvFile.is_open()) return;

    const CompArea& blk = pu.blocks[getFirstComponentOfChannel(pu.chType)];
    uint64_t currentID = m_lineCounter++; // Uso do contador incremental
    
    int w = blk.width;
    int h = blk.height;
    int x = blk.x;
    int y = blk.y;
    int poc = pu.cs->slice->getPOC();

    // cria key única para identificar este bloco específico entre start e end -> garante confiança para futura extração da feature
    std::string key = std::to_string(poc) + "_" + std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(w) + "_" + std::to_string(h) + "_" + std::to_string((int)pu.chType);

    std::stringstream ss;
    ss.imbue(std::locale::classic()); 
    // 1. Metadados e Estatísticas Básicas
    ss << poc << "," << x << "," << y << "," << w << "," << h << "," << baseQP << ","
       << feats.blk_pixel_mean << "," << feats.blk_pixel_variance << "," << feats.blk_pixel_std_dev << "," << feats.blk_pixel_sum << ","
       << feats.blk_var_h << "," << feats.blk_var_v << "," << feats.blk_std_v << "," << feats.blk_std_h << ",";

    // 2. Gradientes
    ss << feats.blk_sobel_gv << "," << feats.blk_sobel_gh << "," << feats.blk_sobel_mag << "," << feats.blk_sobel_dir << "," << feats.blk_sobel_razao_grad << ","
       << feats.blk_prewitt_gv << "," << feats.blk_prewitt_gh << "," << feats.blk_prewitt_mag << "," << feats.blk_prewitt_dir << "," << feats.blk_prewitt_razao_grad << ",";

    // 3. Contraste e Hadamard
    ss << feats.blk_min << "," << feats.blk_max << "," << feats.blk_range << "," << feats.blk_laplacian_var << "," << feats.blk_entropy << ","
       << feats.hadamard.dc << "," << feats.hadamard.energy_total << "," << feats.hadamard.energy_ac << ","
       << feats.hadamard.max_coef << "," << feats.hadamard.min_coef << ","
       << feats.hadamard.top_left << "," << feats.hadamard.top_right << "," << feats.hadamard.bottom_left << "," << feats.hadamard.bottom_right << ",";

    // 4. Geometria 
    ss << CAROL::determine_size_group(w, h) << "," 
       << CAROL::determine_area_group(w, h) << "," 
       << CAROL::determine_orientation_group(w, h) << "," 
       << CAROL::determine_aspect_ratio_group(w, h);

    // armazena no buffer global usando a key (POC_X_Y)
    g_lineBuffer[key] = ss.str();
}

void FeatureLogger::endLine(const CodingUnit& cu) {
    std::lock_guard<std::mutex> lock(g_logMutex);

    // verifica abertura do arquivo csv
    if (!m_csvFile.is_open()) return;

    // recupera a chave usando as coordenadas da CU
    const CompArea& blk = cu.blocks[getFirstComponentOfChannel(cu.chType)];
    std::string key = std::to_string(cu.slice->getPOC()) + "_" + std::to_string(blk.x) + "_" + std::to_string(blk.y) + "_" + std::to_string(blk.width) + "_" + std::to_string(blk.height) + "_" + std::to_string((int)cu.chType);

    // só escreve se houver um início de linha correspondente
    if (g_lineBuffer.find(key) != g_lineBuffer.end()) {
        std::string transName = "UNKNOWN";
        if (cu.rootCbf) {
            switch (cu.firstTU->mtsIdx[COMPONENT_Y]) {
                case MtsType::DCT2_DCT2: transName = "DCT2_DCT2"; break;
                case MtsType::DCT8_DCT8: transName = "DCT8_DCT8"; break;
                case MtsType::DCT8_DST7: transName = "DCT8_DST7"; break;
                case MtsType::DST7_DCT8: transName = "DST7_DCT8"; break;
                case MtsType::DST7_DST7: transName = "DST7_DST7"; break;
                case MtsType::SKIP:      transName = "SKIP";      break;
                default:                 transName = "UNKNOWN";   break;
            }
        }

        // escreve a linha completa de uma vez (de forma atomica)
        m_csvFile << g_lineBuffer[key] << "," << transName << std::endl;

        m_csvFile.flush(); // buffer escrito no disco

        // limpa o buffer para liberar memória 
        g_lineBuffer.erase(key);
    }
}

} 
