/// @file
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min::lib {

    ///	Generate a non-bandlimited <a href="https://en.wikipedia.org/wiki/Sawtooth_wave">sawtooth wave</a> oscillator (a phasor~ in MSP
    ///parlance).
    /// This function is typically used as a control signal for <a href="https://en.wikipedia.org/wiki/Phase_(waves)">phase</a> ramping.

    class sync {
    public:
        /// Set the frequency of the oscillator.
        /// @param	oscillator_frequency	The frequency of the oscillator in hertz.
        /// @param	sampling_frequency		The sampling frequency of the environment in hertz.

        void frequency(number oscillator_frequency, number sampling_frequency) {
            m_fs           = sampling_frequency;
            auto f_nyquist = sampling_frequency * 0.5;
            m_f            = fold(oscillator_frequency, -f_nyquist, f_nyquist);
            m_step         = m_f / m_fs;
        }


        /// Get the current frequency of the oscillator.
        /// @return	The current frequency of the oscillator in the range [0.0, f_s].

        number frequency() {
            return m_f;
        }


        /// Set the phase of the oscillator
        ///	@param	new_phase	The new phase to which the oscillator will be set. Range is [0.0, 1.0).

        void phase(number new_phase) {
            m_phase = wrap(new_phase, 0.0, 1.0);
        }


        /// Get the current phase of the oscillator
        /// @return	The current phase of the oscillator in the range [0.0, 1.0).

        number phase() {
            return m_phase;
        }


        /// Calculate one sample.
        ///	@return		Calculated sample

        sample operator()() {
            if (m_phase >= 1.0)
                m_phase -= 1.0;
            else if (m_phase < 0.0)
                m_phase += 1.0;

            auto y = m_phase;
            m_phase += m_step;
            return y;
        }

    private:
        number m_phase{};    ///< current phase
        number m_step{};     ///< increment for each sample iteration
        number m_f{};        ///< oscillator frequency
        number m_fs{};       ///< sampling frequency
    };


}    // namespace c74::min::lib
