/// @file
///	@ingroup 	minlib
/// @author		Timothy Place
///	@copyright	Copyright (c) 2017, Cycling '74
///	@license	Usage of this file and its contents is governed by the MIT License

#pragma once

namespace c74::min::lib {
    

    ///	Generate an <a href="https://en.wikipedia.org/wiki/Synthesizer#Attack_Decay_Sustain_Release_.28ADSR.29_envelope">ADSR</a> envelope.

    class adsr {
    public:

        enum class adsr_state {
            inactive,
            attack,
            decay,
            sustain,
            release
        };


        class slope {
            const int k_power_multiplier = 5; // higher number yields more extreme curves
        public:

            void operator = (number percentage) {
                m_is_linear = std::abs(percentage) < 0.001;
                if (m_is_linear)
                    m_exp = 1.0;
                else {
                    m_curve = percentage / 100.0;
                    if (m_curve > 0)
                        m_exp = 1.0 + m_curve * k_power_multiplier;
                    else if (m_curve < 0)
                        m_exp = 1.0 + (-m_curve) * k_power_multiplier;
                }
            }

            number operator()(number x) {
                // Profile this: What's worse? branching or calculating even when linear?
                if (m_curve > 0.0)
                    return 1.0 - pow(std::abs(x - 1.0), m_exp);
                else
                    return pow(x, m_exp);
            }

        private:
            number	m_curve		{ 0.0 };
            number	m_exp		{ 1.0 };
            bool	m_is_linear	{ true };
        };


        enum class envelope_mode {
            adsr,
            adr,
            enum_count
        };

        void mode(envelope_mode mode_value) {
            m_envelope_mode = mode_value;
        }

        envelope_mode mode() {
            return m_envelope_mode;
        }


        sample active() {
            return m_state != adsr_state::inactive;
        }


        void initial(number initial_value) {
            m_initial_cached = initial_value;
            recalc();
        }

        void peak(number peak_value) {
            m_peak_cached = peak_value;
            recalc();
        }

        void sustain(number sustain_value) {
            m_sustain_cached = sustain_value;
            recalc();
        }

        void end(number end_value) {
            m_end_cached = end_value;
            recalc();
        }


        /// Set the attack time of the envelope generator.
        /// @param	attack_ms			The attack time in milliseconds.
        /// @param	sampling_frequency	The sampling frequency of the environment in hertz.

        void attack(number attack_ms, number sampling_frequency) {
            m_attack_step_count = static_cast<int>( (attack_ms / 1000.0) * sampling_frequency );
            m_attack_step_count = std::max(m_attack_step_count, 1);
            recalc();
        }


        /// Set the attack slope of the envelope generator.
        /// @param	attack_curve		The attack slope as a +/- percentage.

        void attack_curve(number attack_curve) {
            m_attack_exp = attack_curve;
        }


        /// Set the decay time of the envelope generator.
        /// @param	decay_ms			The decay time in milliseconds.
        /// @param	sampling_frequency	The sampling frequency of the environment in hertz.

        void decay(number decay_ms, number sampling_frequency) {
            m_decay_step_count = static_cast<int>( (decay_ms / 1000.0) * sampling_frequency );
            recalc();
        }


        /// Set the decay slope of the envelope generator.
        /// @param	decay_curve			The decay slope as a +/- percentage.

        void decay_curve(number decay_curve) {
            m_decay_exp = decay_curve;
        }


        /// Set the release time of the envelope generator.
        /// @param	release_ms			The release time in milliseconds.
        /// @param	sampling_frequency	The sampling frequency of the environment in hertz.

        void release(number release_ms, number sampling_frequency) {
            m_release_step_count = static_cast<int>( (release_ms / 1000.0) * sampling_frequency );
            recalc();
        }


        /// Set the release slope of the envelope generator.
        /// @param	release_curve		The release slope as a +/- percentage.

        void release_curve(number release_curve) {
            m_release_exp = release_curve;
        }


        void trigger(bool active) {
            if (active != m_active) {
                m_active = active;

                if (m_active) {
                    m_state = adsr_state::attack;
                    m_index = 0;
                    m_attack_current = 0.0;
                }
                else {
                    m_state = adsr_state::release;
                    m_index = 0;
                    m_release_current = 0.0;
                }
            }
        }


        adsr_state state() {
            return m_state;
        }


        /// Calculate one sample.
        ///	@return		Calculated sample

        sample operator()() {
			sample output {};

            switch (m_state) {
                case adsr_state::attack:
                    m_attack_current += m_attack_step;
                    ++m_index;
                    if (m_index == m_attack_step_count) {
                        output = m_peak_cached;
                        m_state = adsr_state::decay;
                        m_index = 0;
                        m_decay_current = 0.0;
                    }
                    else
                        output = m_attack_exp(m_attack_current) * (m_peak_cached - m_initial_cached) + m_initial_cached;
                    break;
                case adsr_state::decay:
                    m_decay_current += m_decay_step;
                    ++m_index;
                    if (m_index == m_decay_step_count) {
                        output = m_sustain_cached;
                        if (m_envelope_mode == envelope_mode::adsr)
                            m_state = adsr_state::sustain;
                        else
                            m_state = adsr_state::release;
                        m_index = 0;
                        m_release_current = 0;
                    }
                    else
                        output = m_decay_exp(m_decay_current) * (m_sustain_cached - m_peak_cached) + m_peak_cached;
                    break;
                case adsr_state::sustain:
                    output = m_sustain_cached;
                    break;
                case adsr_state::release:
                    m_release_current += m_release_step;
                    ++m_index;
                    if (m_index >= m_release_step_count) {
                        output = m_end_cached;
                        m_state = adsr_state::inactive;
                        m_active = false;
                    }
                    else
                        output = m_release_exp(m_release_current) * (m_end_cached - m_sustain_cached) + m_sustain_cached;
                    break;
                case adsr_state::inactive:
                    output = m_end_cached;
                    break;
            }
            return output;
        }

    private:
        slope	m_attack_exp;
        number	m_attack_step;
        int		m_attack_step_count;
        sample	m_attack_current;

        slope	m_decay_exp;
        number	m_decay_step;
        int		m_decay_step_count;
        sample 	m_decay_current;

        slope	m_release_exp;
        number	m_release_step;
        int		m_release_step_count;
        sample 	m_release_current { 0.0 };

        number	m_initial_cached;
        number	m_peak_cached;
        number	m_sustain_cached;
        number	m_end_cached;

        int	m_index { 0xFFFFFF };

        adsr_state m_state { adsr_state::inactive };

        bool	        m_active { false };
        envelope_mode   m_envelope_mode { envelope_mode::adsr };


        void recalc() {
            m_attack_step = 1.0 / m_attack_step_count;
            m_decay_step = 1.0 / m_decay_step_count;
            m_release_step = 1.0 / m_release_step_count;
        }
    };

    
}  // namespace c74::min::lib
