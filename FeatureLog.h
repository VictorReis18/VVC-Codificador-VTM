#ifndef __FEATURE_LOG_H__
#define __FEATURE_LOG_H__

#include "CommonLib/CodingStructure.h"
#include "CommonLib/Slice.h"
#include "CommonLib/Unit.h"
#include "BlockFeatures.h"
#include "CAROL_GroupFeatures.h"
#include <fstream>
#include <string>

namespace CAROL {

class FeatureLogger {
private:
    std::ofstream m_csvFile;
    bool m_initialized = false;
    std::string m_currentFileName;

    // Construtor privado para Singleton
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
    void endLine(const CodingUnit& cu);

    // Fecha os arquivos manualmente se necessário
    void close() {
        if (m_csvFile.is_open()) m_csvFile.close();
        m_initialized = false;
    }

    // Deletar cópia e atribuição para garantir Singleton puro
    FeatureLogger(const FeatureLogger&) = delete;
    void operator=(const FeatureLogger&) = delete;
};

} // namespace CAROL

#endif
