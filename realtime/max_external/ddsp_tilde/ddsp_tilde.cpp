#include "c74_min.h"

using namespace c74::min;

class ddsp_tilde : public object<ddsp_tilde>, public vector_operator<> {
public:
  MIN_DESCRIPTION{
      "Differentiable Digital Signal Processing wrapper for MAX/MSP"};
  MIN_AUTHOR{"Antoine Caillon"};

  inlet<> f0{this, "(signal) fundamental frequency"};
  inlet<> lo{this, "(signal) loudness"};

  outlet<> out{this, "(signal) synthesized audio", "signal"};

  void operator()(audio_bundle input, audio_bundle output) {
    // ACTUAL GUTS OF THE EXTERNAL
  }
};

MIN_EXTERNAL(ddsp_tilde);