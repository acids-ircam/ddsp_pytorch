/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min::dataspace {

    class time : public dataspace_base {
    public:
        // Seconds is the neutral unit, so it is a pass-through
        class seconds {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                return input;
            }

            static inline number from_neutral(const number input) {
                return input;
            }
        };


        class bark {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                // code from http://labrosa.ee.columbia.edu/matlab/rastamat/bark2hz.m
                return 1.0 / (600 * sinh(double(input) / 6));
            }

            static inline number from_neutral(const number input) {
                // taken from http://labrosa.ee.columbia.edu/matlab/rastamat/hz2bark.m
                return (6 * asinh(1.0 / (double(input) * 600.0)));
            }
        };


        class bpm {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                // TODO: prevent division with zero
                return 60.0 / double(input);
            }

            static inline number from_neutral(const number input) {
                // TODO: prevent division with zero
                return 60.0 / double(input);
            }
        };


        class cents {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                return 1.0 / (440.0 * pow(2.0, (double(input) - 6900.0) / 1200.0));
            }

            static inline number from_neutral(const number input) {
                return 6900.0 + 1200.0 * log(1.0 / (440.0 * double(input))) / log(2.0);
            }
        };


        class hertz {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                // TODO: prevent division with zero
                return 1.0 / double(input);
            }

            static inline number from_neutral(const number input) {
                // TODO: prevent division with zero
                return 1.0 / double(input);
            }
        };


        class mel {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                // HTK-code from http://labrosa.ee.columbia.edu/matlab/rastamat/mel2hz.m
                return 1.0 / (700.0 * (pow(10, (double(input) / 2595.0)) - 1.0));
            }

            static inline number from_neutral(const number input) {
                // HTK-code from http://labrosa.ee.columbia.edu/matlab/rastamat/hz2mel.m
                return 2595.0 * log10(1 + 1.0 / (double(input) * 700.0));
            }
        };


        class midi {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                return 1. / (440.0 * pow(2.0, (double(input) - 69.0) / 12.0));
            }

            static inline number from_neutral(const number input) {
                // return 69.0 + 12.0 * log(1./(440.0*TTFloat64(input)))/log(2.0);
                // The above can be transformed to the slightly more optimised:
                return 69.0 - 12.0 * log(440.0 * double(input)) / log(2.0);
            }
        };


        class milliseconds {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                return input * 0.001;
            }

            static inline number from_neutral(const number input) {
                return input * 1000.0;
            }
        };


        class samples {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                assert(false);    // TODO: Need to get global sample rate
                double sampleRate = 96000;
                return (input) / sampleRate;
            }

            static inline number from_neutral(const number input) {
                assert(false);    // TODO: Need to get global sample rate
                double sampleRate = 96000;
                return (input)*sampleRate;
            }
        };


        class speed {
            friend class dataspace_base;

            static inline number to_neutral(const number input) {
                // Here's one way of converting:
                //
                // TTFloat64 midi;
                // 1) speed => midi
                //		midi = 12.0 * log(TTFloat64(input))/log(2.0);
                // 2) midi => second
                //		output = 1. / (440.0 * pow(2.0, (midi-69.0) / 12.0 ));

                // This is an optimized version of the above:
                return pow(2.0, 69. / 12.) / (440.0 * double(input));
            }

            static inline number from_neutral(const number input) {
                // Here's one way of converting from second to speed:
                //
                // TTFloat64 midi;
                // 1) second => midi
                //		midi = 69.0 - 12.0 * log(440.0*TTFloat64(input))/log(2.0);
                // 2) midi => speed
                //		output = pow(2.0, (midi/12.0));

                // Optimized in a similar way to the above:
                return pow(2.0, 69. / 12.) / (440.0 * double(input));
            }
        };
    };

}    // namespace c74::min::dataspace
