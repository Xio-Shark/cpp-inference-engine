#if defined(USE_CUDA) && !defined(__APPLE__)
// Legacy transformer.cu for direct CUDA/cuBLAS compilation
#include "transformer.h"
#include "kernels.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
// Implementation moved to cross-platform transformer.cpp
#endif
