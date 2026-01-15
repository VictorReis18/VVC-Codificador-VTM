#ifndef __FEATURE_LOG_H__
#define __FEATURE_LOG_H__

#include "CommonLib/CodingStructure.h"
#include "CommonLib/Slice.h"
#include "CommonLib/Unit.h"
#include "BlockFeatures.h"
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <atomic>

namespace CAROL {

class FeatureLogger {
private:
    std::ofstream m_csvFile;
    bool m_initialized = false;
    std::atomic<uint64_t> m_lineCounter{0}; // Contador

    // Construtor privado
    FeatureLogger() {}

public:
    // Retorna a instância única do Logger
    static FeatureLogger& getInstance() {
        static FeatureLogger instance;
        return instance;
    }

    // Inicializa o arquivo CSV com base no nome do input e QP
    void init(const std::string& inputName, int qp);

    // Escreve a primeira parte da linha (Features + Geometria)
    void startLine(const PredictionUnit& pu, const BlockFeatures& feats, int qp);

    // Escreve a parte final (Transformada) e quebra a linha
    void endLine(const CodingUnit& cu,const std::string& key);

    // Fecha os arquivos manualmente se necessário
    void close() {
        if (m_csvFile.is_open()) m_csvFile.close();
        m_initialized = false;
    }

    // Deletar cópia e atribuição para garantir Singleton
    FeatureLogger(const FeatureLogger&) = delete;
    void operator=(const FeatureLogger&) = delete;
};

// Funções fornecidas em Python para extração de features do grupo
// 1. Determina o grupo baseado na maior dimensão
    inline int determine_size_group(int w, int h) {
        int max_dim = std::max(w, h);
        if (max_dim >= 128) return 128; 
        if (max_dim == 64)  return 64;
        if (max_dim == 32)  return 32;
        if (max_dim == 16)  return 16;
        if (max_dim == 8)   return 8;
        return 4; // Para 4 e outros menores
    }

    // 2. Determina a área do bloco
    inline int determine_area_group(int w, int h) {
        return std::min(w, h) * std::max(w, h);
    }

    // 3. Determina orientação (0: Quadrado, 1: Horizontal, 2: Vertical)
    inline int determine_orientation_group(int w, int h) {
        if (w == h) return 0; // square
        return (w > h) ? 1 : 2; // horizontal
    }

    // 4. Determina a proporção (Aspect Ratio)
    inline int determine_aspect_ratio_group(int w, int h) {
        
        double ratio = (double)std::max(w, h) / std::min(w, h);
        
        if (std::abs(ratio - 1.0) < 0.01)  return 0; // 1:1
        if (std::abs(ratio - 2.0) < 0.01)  return 1; // 2:1
        if (std::abs(ratio - 4.0) < 0.01)  return 2; // 4:1
        if (std::abs(ratio - 8.0) < 0.01)  return 3; // 8:1
        if (std::abs(ratio - 16.0) < 0.01) return 4; // 16:1
        if (std::abs(ratio - 32.0) < 0.01) return 5; // 32:1
        return 6; // Outro
    }

}

#endif
