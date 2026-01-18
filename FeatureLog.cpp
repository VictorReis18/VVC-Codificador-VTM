#include "FeatureLog.h"
#include "CommonLib/CodingStructure.h"
#include "CommonLib/Slice.h"
#include "CommonLib/Unit.h"
#include <map>
#include <mutex>
#include <sstream>
#include <locale>
#include <vector>
#include <random>
#include <fstream>

namespace CAROL {

// Estrutura global para evitar conflitos entre threads 
static std::map<std::string, std::string> g_lineBuffer;
static std::mutex g_logMutex;

// Globals for reservoir sampling
static std::map<std::string, std::vector<std::string>> g_reservoirs;
static std::map<std::string, uint64_t> g_counts;
static std::string g_videoName;
static int g_qp = 0;
static const size_t RESERVOIR_SIZE = 7000;
static std::mt19937_64 g_rng(std::random_device{}());

// Flusher to write files at exit
struct ReservoirFlusher {
    ~ReservoirFlusher() {
        if (g_videoName.empty()) return;

        std::string header = "POC,X,Y,W,H,QP,"
                  "Mean,Var,StdDev,Sum,VarH,VarV,StdV,StdH,"
                  "SobelGV,SobelGH,SobelMag,SobelDir,SobelRatio,"
                  "PrewittGV,PrewittGH,PrewittMag,PrewittDir,PrewittRatio,"
                  "Min,Max,Range,LaplacianVar,Entropy,"
                  "H_DC,H_EnergyTotal,H_EnergyAC,H_Max,H_Min,"
                  "H_TL,H_TR,H_BL,H_BR,"
                  "SizeGroup,Area,Orientation,AspectRatioIdx,"
                  "Resi_SAD,Resi_LastRowSum,Resi_LastColSum,Resi_TL,Resi_TR,Resi_BR,"                  
                  "Transformada";

        for (auto const& [blockSize, lines] : g_reservoirs) {
            std::string fileName = g_videoName + "-" + std::to_string(g_qp) + "-" + blockSize + ".csv";
            std::ofstream outFile(fileName);
            if (outFile.is_open()) {
                outFile << header << std::endl;
                for (const auto& line : lines) {
                    outFile << line << std::endl;
                }
                outFile.close();
            }
        }
    }
};

static ReservoirFlusher g_flusher;

void FeatureLogger::init(const std::string& inputName, int qp) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    if (m_initialized) return;

    g_videoName = inputName;
    g_qp = qp;
    m_initialized = true;
}

std::string FeatureLogger::startLine(const PredictionUnit& pu, const BlockFeatures& feats, int baseQP) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    
    if (!m_initialized) return "";

    const CompArea& blk = pu.blocks[getFirstComponentOfChannel(pu.chType)];
    uint64_t currentID = m_lineCounter++; // Uso do contador incremental

    int w = blk.width;
    int h = blk.height;
    int x = blk.x;
    int y = blk.y;
    int poc = pu.cs->slice->getPOC();

    // cria key única para identificar este bloco específico entre start e end -> garante confiança para futura extração da feature
    std::string key = std::to_string(pu.cs->slice->getPOC()) + "_" + 
                        std::to_string(blk.x) + "_" + 
                        std::to_string(blk.y) + "_" + 
                        std::to_string(blk.width) + "_" + 
                        std::to_string(blk.height) + "_" + 
                        std::to_string((int)pu.chType) + "_" + 
                        std::to_string(currentID);
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
    return key;
}

void FeatureLogger::endLine(const CodingUnit& cu) {
    std::lock_guard<std::mutex> lock(g_logMutex);

    // Recupera a chave
    const std::string& key = cu.carolKey;
    // verifica abertura do arquivo csv
    if (!m_initialized) return;

    // só escreve se houver um início de linha correspondente
    if (!key.empty() && g_lineBuffer.find(key) != g_lineBuffer.end()) {
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

        // Constrói a linha completa
        std::string fullLine = g_lineBuffer[key] + "," + transName;

        // Determina o tamanho do bloco para o reservatório
        const CompArea& blk = cu.blocks[getFirstComponentOfChannel(cu.chType)];
        std::string blockSize = std::to_string(blk.width) + "x" + std::to_string(blk.height);

        // Amostragem de Reservatório
        uint64_t& count = g_counts[blockSize];
        count++;

        if (g_reservoirs[blockSize].size() < RESERVOIR_SIZE) {
            g_reservoirs[blockSize].push_back(fullLine);
        } else {
            std::uniform_int_distribution<uint64_t> dist(0, count - 1);
            uint64_t j = dist(g_rng);
            if (j < RESERVOIR_SIZE) {
                g_reservoirs[blockSize][j] = fullLine;
            }
        }

        // limpa o buffer para liberar memória 
        g_lineBuffer.erase(key);
    }
}

} 
