#pragma once
#include <torch/torch.h>
#include <torch/script.h>

#define PARTIAL_NUMBER 100
#define FILTER_SIZE    81


class DDSPCore {
public:
  DDSPCore();
  void getNextOutput(float f0, float lo, float * result);

protected:
  torch::jit::script::Module m_module;
  torch::Tensor m_f0, m_lo, m_hx;
};
