#include "c74_min.h"

using namespace c74::min;

class ddsp_tilde : public object<ddsp_tilde>, public sample_operator<1, 1> {
public:
  MIN_DESCRIPTION{"Divide signal by 2"};
  MIN_AUTHOR{"Antoine Caillon"};

  inlet<> in1{this, "(signal) Input 1"};

  outlet<> out{this, "(signal) Output 1", "signal"};

  sample operator()(sample input) {
    // ACTUAL GUTS OF THE EXTERNAL
    return input / 2.;
  }
};

MIN_EXTERNAL(ddsp_tilde);