/// @file
///	@brief 		Unit test for the dcblocker class
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#define CATCH_CONFIG_MAIN
#include "c74_min_catch.h"


SCENARIO ("produce the correct impulse response") {

    GIVEN ("An instance of the dcblocker class") {
        c74::min::lib::dcblocker f;

        WHEN ("processing a 64-sample impulse") {

            // create an impulse buffer to process
            const int				buffersize = 64;
            c74::min::sample_vector	impulse(buffersize);

            std::fill_n(impulse.begin(), buffersize, 0.0);
            impulse[0] = 1.0;

            // output from our object's processing
            c74::min::sample_vector	output;

            // run the calculations
            for (auto x : impulse) {
                auto y = f(x);
                output.push_back(y);
            }

            // get a reference impulse response to compare against
            auto reference = c74::min::lib::filters::generate_impulse_response({1.0,-1.0}, {1.0,-0.9997}, buffersize);

            THEN("The result produced matches an externally produced reference impulse")
            REQUIRE( output == reference );
        }
    }
}
