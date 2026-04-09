import math

def get_code(model, feature_names, class_names, function_name="logistic_model"):
    coefficients = model.coef_[0]
    bias = model.intercept_[0]
    
    code = ""
    code += f"\t// Bias (Intercept)\n"
    code += f"\tdouble score = {bias:.16f};\n\n"
    
    code += f"\t// Weighted Sum (Dot Product)\n"
    for i, (coef, feat_name) in enumerate(zip(coefficients, feature_names)):
        if abs(coef) > 1e-9:
            code += f"\tscore += feature_vector.at({i}) * {coef:.16f}; // {feat_name}\n"
    
    code += "\n\t// Return Probability using Sigmoid: 1 / (1 + exp(-score))\n"
    code += "\treturn 1.0 / (1.0 + std::exp(-score));\n"
    wrapper = f"inline double {function_name}(const std::vector<double> & feature_vector) \n{{\n{code}}}"
    return wrapper

def save_code(model, feature_names, class_names, function_name="logistic_model", output_dir="."):
    feature_string = ""
    for i in range(len(feature_names)):
        feature_string += f"feature_vector[{i}] - {feature_names[i]}\n"
    
    preamble = f"""
/*
This inline function was automatically generated using LogisticRegToCpp Converter
It takes feature vector as single argument:
{feature_string}

It returns the PROBABILITY (double) of the positive class (e.g., IsIntra).
Range: [0.0, 1.0]
*/
"""
    cpp_logic = get_code(model, feature_names, class_names, function_name)
    
    final_content = f"{preamble}#include <vector>\n#include <cmath>\n\n{cpp_logic}\n"

    filename = f"{output_dir}/{function_name}.h"
    with open(filename, "w") as f:
        f.write(final_content)
    
    return 0