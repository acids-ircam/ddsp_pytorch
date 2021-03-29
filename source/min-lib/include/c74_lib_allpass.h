/// @file
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_lib_delay.h"

namespace c74::min::lib {

    
    ///	A single-channel generalized allpass filter.

    class allpass {
    public:
        /// Default constructor with minimum number of initial values.
        /// @param	initial_size	Sets initial delay size in samples for feedforward and feedback history.
        ///							Default value is 4410 samples. Capacity is fixed at creation.
        /// @param	initial_gain	Sets the gain coefficient that is applied to samples from history.
        ///							Default value is 0.0.

        explicit allpass(number initial_size = 4410, number initial_gain = 0.0)
        : m_feedforward_history(initial_size)
        , m_feedback_history(initial_size) {
            this->gain(initial_gain);
        }


        /// Constructor with initial values for capacity, size, and gain.
        /// @param  capacity_and_size	Sets capacity and size in samples for feedforward and feedback history.
        ///								Uses std::pair to ensure values are set together. Capacity is fixed at creation.
        /// @param	initial_gain		Sets the gain coefficient that is applied to samples from history.
        ///								Default value is 0.0.

        explicit allpass(std::pair<size_t, number> capacity_and_size, number initial_gain = 0.0)
        : m_feedforward_history(capacity_and_size)
        , m_feedback_history(capacity_and_size) {
            this->gain(initial_gain);
        }


        /// Set a new delay time in samples.
        /// @param	new_size	The new delay time in samples.
        ///						Note this may not exceed the capacity set when the object instance is created.

        void delay(number new_size) {
            m_feedforward_history.size(new_size);
            m_feedback_history.size(new_size);
        };


        /// Return the current delay time in samples.
        /// @return The delay time in samples.

        number delay() {
            return m_feedforward_history.size();
        };


        /// Set a new delay time in milliseconds.
        /// @param	new_size_ms		The new delay time in milliseconds.
        /// @param	sampling_frequency		The sampling frequency of the environment in hertz.

        void delay_ms(number new_size_ms, number sampling_frequency) {
            delay(math::milliseconds_to_samples(new_size_ms,sampling_frequency));
        }


        /// Set the feedback coefficient.
        /// @param	new_gain	The new feedback coefficient.

        void gain(number new_gain) {
            m_gain = new_gain;
        }


        /// Return the value of the feedback coefficient
        /// @return The current feedback coefficient.

        number gain() {
            return m_gain;
        }


        ///	This algorithm is an IIR filter, meaning that it relies on feedback.  If the filter should
        ///	not be producing any signal (such as turning audio off and then back on in a host) or if the
        ///	feedback has become corrupted (such as might happen if a NaN is fed in) then it may be
        ///	neccesary to clear the filter by calling this method.

        void clear() {
            m_feedforward_history.clear();
            m_feedback_history.clear();
        }


        /// Change the interpolation algorithm used.
        /// @param	new_type	option from the interpolator::type enum that names algorithm

        void change_interpolation(interpolator::type new_type = interpolator::type::none) {
            m_feedforward_history.change_interpolation(new_type);
            m_feedback_history.change_interpolation(new_type);
        }


        /// Calculate one sample.
        ///	@return		Calculated sample

        sample operator()(sample x) {
            auto x1    = m_feedforward_history.tail(1);
            auto y1    = m_feedback_history.tail(1);
            auto alpha = m_gain;

            // Store the input in the feedforward buffer
            m_feedforward_history.write(x);

            // Apply the filter
            // We start with the equation in standard form:
            //		y = -alpha * x  +  x1  +  alpha * y1;
            // Then to a version that Fred Harris refers to as a "Re-Ordered All-Pass Filter Structure" in Multirate Signal Processing
            //		y = x1  +  alpha * y1  -  alpha * x;
            // Finally, here is a "Single Coefficient All-Pass Filter", dropping from 2 adds and 2 mults down to 2 adds and 1 mult
            auto y = x1 + ((y1 - x) * alpha);

            // Store the output in the feedback buffer
            m_feedback_history.write(y);

            return y;
        }

    private:
        c74::min::lib::delay m_feedforward_history{};    ///< Delay line for the FIR side of the filter.
        c74::min::lib::delay m_feedback_history{};       ///< Delay line for the IIR side of the filter.
        number               m_gain{};                   ///< Feedback coefficient.
    };


}    // namespace c74::min::lib
