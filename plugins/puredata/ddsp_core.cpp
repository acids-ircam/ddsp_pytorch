#include "ddsp_core.hpp"

DDSPCore::DDSPCore() {
  try {
    m_module = torch::jit::load("ddsp.torchscript");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  std::cout << "Model successfully loaded!\n";

  m_f0 = torch::zeros({1,1,1});
  m_lo = torch::zeros({1,1,1});
  m_hx = torch::zeros({1,1,512});
  
}

void DDSPCore::getNextOutput(float f0, float lo, float * result){
  m_f0[0][0][0] = f0;
  m_lo[0][0][0] = lo;

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(m_f0);
  inputs.push_back(m_lo);
  inputs.push_back(m_hx);

  auto output = m_module.forward(inputs).toTuple()->elements();

  auto amp    = output[0].toTensor();
  auto alpha  = output[1].toTensor();
  auto filter = output[2].toTensor();
  auto hx     = output[3].toTensor();

  m_hx = hx;

  result[0] = amp[0][0][0].item<float>();

  for (int i(0); i<PARTIAL_NUMBER; i++) {
    result[i+1] = alpha[0][0][i].item<float>();
  }

  for (int i(0); i<FILTER_SIZE; i++) {
    result[i+1+PARTIAL_NUMBER] = filter[0][0][i].item<float>();
  }

}
