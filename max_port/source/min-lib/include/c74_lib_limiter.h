/// @file
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_min_api.h"


namespace c74::min::lib {


    ///	Lookahead limiter for n-channels of audio.

    class limiter {
    public:


        /// Create a limiter instance.
        /// @param a_channel_count	The number of channels to process with the limiter.
        /// @param a_buffer_size	The maximum number of samples that may be used for the lookahead function.
        /// @param a_samplerate		The samplerate at which the limiter will operate.
        ///							The samplerate may be updated at a later time by calling the reset() method.

        limiter(int a_channel_count = 2, int a_buffer_size = 512, number a_samplerate = 48000.0) {
            m_buffer_size = a_buffer_size;
            m_channelcount = a_channel_count;
            m_samplerate = a_samplerate;

            for (auto i = 0; i < m_channelcount; ++i)
                m_dcblockers.push_back(std::make_unique<lib::dcblocker>());

            m_lookahead_buffers.resize(m_channelcount);
            for (auto& buffer : m_lookahead_buffers)
                buffer.resize(m_buffer_size);
            m_gain_buffer.resize(m_buffer_size);
            clear();
        }

#if 0
#pragma mark -
#pragma mark attributes
#endif

        /// Set the bypass state.
        /// @param	a_state	The new state of the limiter bypass.

        void bypass(bool a_state) {
            m_bypass = a_state;
        }

        /// Return the current state of the limiter bypass.
        /// @return The current state of the limiter bypass.

        number bypass() {
            return m_bypass;
        }


        /// Turn on/off the dc-offset blocking filter applied to the input.
        /// @param	use_dc_blocker	The new active state of the dc-blocker.

        void dcblock(bool use_dc_blocker) {
            m_dcblock = use_dc_blocker;
        }

        /// Return the current state of the dc-offset blocking filter applied to the input.
        /// @return The current active state of the dcblocker.

        number dcblock() {
            return m_dcblock;
        }


        /// Options for how the release is handled in the limiter.

        enum class response_mode : int {
            linear,			///< linear
            exponential,	///< exponential
            enum_count
        };

        /// Set the shape used for the release response in the limiter.
        /// @param	a_mode	The new shape for the mode.

        void mode(response_mode a_mode) {
            m_mode = a_mode;
            reset(m_release, m_mode, m_samplerate);
        }

        /// Return the current mode of the limiter.
        /// @return The current mode of the limiter.

        response_mode mode() {
            return m_mode;
        }


        /// Number of samples to look ahead.
        /// @param sample_count_for_lookahead_buffer	The number of samples to look ahead.
        ///												Must be less than or equal to the buffer size specified at creation time.

        void lookahead(int sample_count_for_lookahead_buffer) {
            m_lookahead = sample_count_for_lookahead_buffer;
            m_lookahead_inv = 1.0 / static_cast<number>(m_lookahead);
        }

        /// Return the number of samples currently used for the lookahead function.
        /// @return The current lookahead value.

        int lookahead() {
            return m_lookahead;
        }


        /// Gain (db) applied prior to processing.
        /// @param gain_in_db New value in decibels.

        void preamp(number gain_in_db) {
            using namespace dataspace;

            m_preamp = gain_in_db;
            m_linear_preamp = gain::convert<gain::db, gain::linear>(m_preamp);
        }

        /// Gain (db) applied prior to processing.
        /// @return The current value in decibels.

        number preamp() {
            return m_preamp;
        }


        /// Gain (db) applied after to processing.
        /// @param gain_in_db New value in decibels.

        void postamp(number gain_in_db) {
            using namespace dataspace;

            m_postamp = gain_in_db;
            m_linear_postamp = gain::convert<gain::db, gain::linear>(m_postamp);
        }

        /// Gain (db) applied after to processing.
        /// @return The current value in decibels.

        number postamp() {
            return m_postamp;
        }


        /// Level (db) above which to apply limiting.
        /// @param threshold_in_db New value in decibels.

        void threshold(number threshold_in_db) {
            using namespace dataspace;

            m_threshold = threshold_in_db;
            m_linear_threshold = gain::convert<gain::db, gain::linear>(m_threshold);
        }

        /// Level (db) above which to apply limiting.
        /// @return The current value in decibels.

        number threshold() {
            return m_threshold;
        }


        /// Millisecond release time.
        /// @param release_time_in_ms New release time.

        void release(number release_time_in_ms) {
            m_release = release_time_in_ms;
            reset(m_release, m_mode, m_samplerate);
        }

        /// Millisecond release time.
        /// @return The current value in milliseconds.

        number release() {
            return m_release;
        }

#if 0
#pragma mark -
#pragma mark methods
#endif

        /// Reset the limiter history.

        void clear() {
            for (auto& filter : m_dcblockers)
                filter->clear();
            for (auto& buffer : m_lookahead_buffers)
                std::fill(buffer.begin(), buffer.end(), 0.0);
            std::fill(m_gain_buffer.begin(), m_gain_buffer.end(), 1.0);
            m_last = 1.0;
            m_lookahead_index = 0;

            reset(m_release, m_mode, m_samplerate);
        }


        /// Reset time-dependent internal coefficients

        void reset(number release, response_mode mode, number sample_rate) {
            m_recover = 1000.0 / (release * sample_rate);
            if (mode == response_mode::linear)
                m_recover *= 0.5;
            else // exponential
                m_recover *= 0.707;
        }

#if 0
#pragma mark -
#pragma mark audio
#endif

        /// Calculate n-samples for m-channels.
        /// The number of channels at the input and output must match the channel count of the limiter.

        void operator()(audio_bundle input, audio_bundle output) {
            if (m_bypass) {
                output = input;
                return;
            }

            int    lookahead = m_lookahead;
            bool   is_linear = (m_mode == response_mode::linear);
            bool   dcblock = m_dcblock;
            sample v;

            for (auto i = 0; i < input.frame_count(); ++i) {
                sample hot_sample {0.0};

                for (auto channel = 0; channel < m_channelcount; ++channel) {
                    auto x = input.samples(channel);

                    // Preprocessing (DC Blocking, Preamp)

                    v = dcblock ? (*m_dcblockers[channel])(x[i]) : x[i];
                    v *= m_linear_preamp;

                    // Analysis

                    m_lookahead_buffers[channel][m_lookahead_index] = v * m_linear_postamp;
                    v = fabs(v);
                    if (v > hot_sample)
                        hot_sample = v;
                }

                if (is_linear)
                    v = m_last + m_recover;
                else {
                    if (m_last > 0.01)
                        v = m_last + m_recover * m_last;
                    else
                        v = m_last + m_recover;
                }

                if (v > 1)
                    v = 1;
                m_gain_buffer[m_lookahead_index] = v;

                int lookahead_playback = m_lookahead_index - lookahead;
                if (lookahead_playback < 0)
                    lookahead_playback += lookahead;

                // Process

                if (hot_sample * v > m_linear_threshold) {
                    number newgain;
                    auto   curgain = m_linear_threshold / hot_sample;
                    auto   inc     = m_linear_threshold - curgain;
                    auto   acc     = 0.0;
                    auto   flag    = 0;

                    for (auto j = 0; flag == 0 && j < lookahead; j++) {
                        auto k = m_lookahead_index - j;

                        if (k < 0)
                            k += lookahead;

                        if (is_linear)
                            newgain = curgain + inc * acc;
                        else
                            newgain = curgain + inc * (acc * acc);

                        if (newgain < m_gain_buffer[k])
                            m_gain_buffer[k] = newgain;
                        else
                            flag = 1;
                        acc = acc + m_lookahead_inv;
                    }
                }

                // Apply Gain

                for (auto channel = 0; channel < m_channelcount; ++channel) {
                    auto y = output.samples(channel);
                    y[i]   = m_lookahead_buffers[channel][lookahead_playback] * m_gain_buffer[lookahead_playback];
                }

                m_last = m_gain_buffer[m_lookahead_index];
                ++m_lookahead_index;
                if (m_lookahead_index >= lookahead)
                    m_lookahead_index = 0;
            }
        }


    private:
        int									m_channelcount		{};  	  	// number of channels
        int									m_buffer_size		{};
        number								m_samplerate		{48000};
        vector<unique_ptr<lib::dcblocker>>	m_dcblockers;
        bool								m_dcblock			{true};
        bool								m_bypass			{false};
        response_mode						m_mode				{response_mode::exponential};
        number								m_preamp			{0.0};		// in db
        number								m_linear_preamp		{1.0};
        number								m_postamp			{0.0};		// in db
        number								m_linear_postamp	{1.0};
        number								m_threshold			{0.0};		// in db
        number								m_linear_threshold	{1.0};
        number								m_release	 		{1000.0};	// in ms
		number                              m_recover {0.0};
        number								m_last				{0.0};
        int									m_lookahead			{100};		// in samples
        number								m_lookahead_inv		{1/100};
        int									m_lookahead_index	{0};
        vector<sample_vector>				m_lookahead_buffers;
        sample_vector						m_gain_buffer;
    };

}    // namespace c74::min::lib
