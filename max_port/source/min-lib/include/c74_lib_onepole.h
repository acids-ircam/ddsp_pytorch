/// @file
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once


namespace c74::min::lib {


    ///	Single-channel, basic <a href="https://en.wikipedia.org/wiki/Low-pass_filter">low-pass filter</a>.
    ///	Handy for use in smoothing control signals or damping high frequency components.

    class onepole {
    public:
        /// Default constructor with minimum number of initial values.
        /// @param	initial_coefficient		Sets the gain coefficient that is applied to samples from history.
        ///									Default value is 0.5.

        explicit onepole(number initial_coefficient = 0.5) {
            this->coefficient(initial_coefficient);
        }


        /// Set filter coefficient directly.
        /// @param new_coefficient	The new value of the feedback coefficient in the range [0.0, 1.0].

        void coefficient(number new_coefficient) {
            new_coefficient = MIN_CLAMP(new_coefficient, 0.0, 1.0);
            b_1             = new_coefficient;
            a_0             = 1 - new_coefficient;
        }

        /// Get the current coefficent of the filter.
        /// @return	The value of the feedback coefficient.

        number coefficient() {
            return b_1;
        }


        /// Set filter coefficient using a cutoff frequency.
        /// @param cutoff_frequency		The cutoff frequency in hertz.
        /// @param sampling_frequency	The sample frequency in hertz.
        ///	@see http://musicdsp.org/showArchiveComment.php?ArchiveID=237

        void frequency(number cutoff_frequency, number sampling_frequency) {
            coefficient(1.0 - exp(-2.0 * M_PI * cutoff_frequency / sampling_frequency));
        }


        /// Clear the filter's history

        void clear() {
            y_1 = 0.0;
        }


        /// Retrieve the filter's history
        /// @return	The value stored in the filter's history.

        sample history() {
            return y_1;
        }


        /// Calculate one sample.
        ///	@return		Calculated sample

        sample operator()(sample x) {
            auto y = (x * a_0) + (y_1 * b_1);
            y_1    = y;
            return y;
        }

    private:
        number a_0{0.5};    ///< gain coefficient
        number b_1{0.5};    ///< feedback coefficient
        sample y_1{};       ///< previous output sample
    };

}    // namespace c74::min::lib
