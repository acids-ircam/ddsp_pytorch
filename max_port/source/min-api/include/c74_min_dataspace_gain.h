/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min::dataspace {

    class gain : public dataspace_base {
    public:
        // Linear is the neutral unit, so it is a pass-through
        class linear {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                return input;
            }

            static inline number from_neutral(const number input) {
                return input;
            }
        };


        class midi {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                return pow(input * 0.01, k_gain_midi_power);
            }

            static inline number from_neutral(const number input) {
                return 100.0 * pow(input, k_gain_midi_power_r);
            }
        };


        class db {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                return pow(10.0, input * 0.05);
            }

            static inline number from_neutral(const number input) {
                number temp = log10(input) * 20.0;

                // Output decibel range is limited to 24 bit range, avoids problems with singularities (-inf) when using
                // dataspace in ramps
                if (temp < -144.49)
                    temp = -144.49;
                return temp;
            }
        };
    };

}    // namespace c74::min::dataspace
