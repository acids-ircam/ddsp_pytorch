/// @file
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

const int k_total_padding = 8;
const int k_half_padding = k_total_padding / 2;

namespace c74::min::lib {

    /// Generate basic <a href="https://en.wikipedia.org/wiki/Waveform">waveforms</a> using a single-cycle
    /// <a href="https://en.wikipedia.org/wiki/Wavetable_synthesis">wavetable</a>
    /// @tparam initial_waveform_type	generator option defined in c74_lib_generator.h

    template<class initial_waveform_type = generator::sine<>>
    class oscillator {
    public:
        /// Constructor specifies a fixed size for the internal wavetable.
        ///	If you want a different size, create a new oscillator and dispose of the one you don't want.
        /// @param	wavetable_size	The number of samples in the wavetable.

        oscillator(std::size_t wavetable_size = 4096)
        : m_wavetable(wavetable_size + k_total_padding) {	// extra samples to accomodate the padding at beginning and end
            change_waveform<initial_waveform_type>();
        }


        ///	Return the size of the internal wavetable without padding.
        /// @return	The size of the sample_vector containing our single-cycle wavetable.

        std::size_t size() {
            return m_wavetable.size() - k_total_padding;
        }


        /// Set the frequency of the oscillator.
        /// @param	oscillator_frequency	The frequency of the oscillator in hertz.
        /// @param	sampling_frequency		The sampling frequency of the environment in hertz.

        void frequency(number oscillator_frequency, number sampling_frequency) {
            m_phase_ramp.frequency(oscillator_frequency, sampling_frequency);
        }


        /// Get the current frequency of the oscillator.
        /// @return	The current frequency of the oscillator in the range [0.0, f_s].

        number frequency() {
            return m_phase_ramp.frequency();
        }


        /// Set the phase of the oscillator
        ///	@param	new_phase	The new phase to which the oscillator will be set. Range is [0.0, 1.0).

        void phase(number new_phase) {
            m_phase_ramp.phase(new_phase);
        }


        /// Get the current phase of the oscillator
        /// @return	The current phase of the oscillator in the range [0.0, 1.0).

        number phase() {
            return m_phase_ramp.phase();
        }


        /// Generate a new shape for the internal wavetable.
        /// @tparam new_waveform_type	generator object defined in c74_lib_generator.h

        template<class new_waveform_type = generator::sine<>>
        void change_waveform() {
            // first generate one cycle inside the padding
            std::generate(m_wavetable.begin()+k_half_padding, m_wavetable.end()-k_half_padding, new_waveform_type(size()));

            // second copy samples to the padding
            std::copy(m_wavetable.end()-k_total_padding, m_wavetable.end()-k_half_padding, m_wavetable.begin());
            std::copy(m_wavetable.begin()+k_half_padding, m_wavetable.begin()+k_total_padding, m_wavetable.end()-k_half_padding);
        }


        /// Return the value at an interpolated position.
        /// @param	new_position	Position within the sample_vector. Non-integer values will be interpolated.
        /// @return Interpolated sample value.

        sample at(number new_position) {
            std::size_t position_integral = static_cast<std::size_t>(new_position);
            number position_fractional = new_position - position_integral;

            return m_interpolator(	m_wavetable.at(position_integral-1),
                                    m_wavetable.at(position_integral),
                                    m_wavetable.at(position_integral+1),
                                    m_wavetable.at(position_integral+2),
                                    position_fractional	);

        }


        /// Calculate one sample.
        ///	@return		Calculated sample

        sample operator()() {
            sample phase_now    = m_phase_ramp();
            sample position_now = phase_now * size() + k_half_padding;

            return at(position_now);
        }


    private:
        sync					m_phase_ramp{};    ///< manages the frequency and phase of our oscillator
        sample_vector			m_wavetable{};     ///< vector containing single cycle of the waveform
        interpolator::cubic<>   m_interpolator{};    ///< The interpolator instance used to produce interpolated output.
    };

}    // namespace c74::min::lib
