/// @file
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_min_api.h"

namespace c74::min::lib {

    
    /// one-channel dc-blocking filter

    class dcblocker {
    public:
        /// Clear the filter's history

        void clear() {
            x_1 = y_1 = 0.0;
        }


        /// Calculate one sample.
        ///	@return		Calculated sample

        sample operator()(sample x) {
            auto y = x - x_1 + y_1 * 0.9997;
            y_1    = y;
            x_1    = x;
            return y;
        }

    private:
        sample x_1{};    ///< feedforward history
        sample y_1{};    ///< feedback history
    };


}   // namespace c74::min::lib
