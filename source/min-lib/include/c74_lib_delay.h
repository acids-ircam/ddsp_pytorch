/// @file
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min::lib {


    ///	A single-channel interpolating delay line.

    class delay {
    public:
        /// Default constructor with minimum number of initial values.
        /// @param	initial_size	Sets initial delay size in samples.
        ///		Because capacity of the delay is fixed at instantiation, this will also be maximum delay allowed.
        ///		Default is 256 samples.

        delay(number initial_size = 256)
        : m_history(static_cast<size_t>(initial_size + 5))    // 5 extra samples to accomodate the 'now' sample + up to 4 interpolation samples
        {
            size(initial_size);
        }


        /// Constructor with initial values for capacity and size.
        /// @param capacity_and_size	Sets capacity and size in samples for delay history.
        ///			Uses std::pair to ensure values are set together. Capacity is fixed at creation.
        ///			First value (capacity) must be greater than the second value (size).

        delay(std::pair<size_t, number> capacity_and_size)
        : m_history(capacity_and_size.first + 5) {
            assert(capacity_and_size.first > capacity_and_size.second);
            size(capacity_and_size.second);
        }


        /// Set a new delay time in samples.
        /// @param	new_size	The new delay time in samples.

        void size(number new_size) {
            m_size            = new_size;
            m_size_integral   = static_cast<std::size_t>(m_size);
            m_size_fractional = m_size - m_size_integral;
        }

        /// Set a new delay time in samples.
        /// @param	new_size	The new delay time in samples.

        void size(size_t new_size) {
            m_size = static_cast<number>(new_size);
            m_size_fractional = 0;
        }


        /// Set a new delay time in samples.
        /// @param	new_size	The new delay time in samples.

        void size(int new_size) {
            m_size = static_cast<number>(new_size);
            m_size_fractional = 0;
        }


        /// Return the current delay time in samples.
        /// @return The delay time in samples.

        number size() const {
            return m_size;
        }


        /// Set a new delay time in milliseconds.
        /// @param	new_size_ms		The new delay time in milliseconds.
        /// @param	sampling_frequency		The sampling frequency of the environment in hertz.

        void size_ms(number new_size_ms, number sampling_frequency) {
            size(math::milliseconds_to_samples(new_size_ms,sampling_frequency));
        }


        /// Return the integer part of the current delay time in samples.
        /// @return The integer part of the delay time in samples.

        std::size_t integral_size() {
            return m_size_integral;
        }


        /// Return the fractional part of the current delay time in samples.
        /// @return The fractional part of the delay time in samples.

        double fractional_size() {
            return m_size_fractional;
        }


        /// Read a single sample out from the delay.
        ///	This will be the oldest sample in the history, unless an offset is specified.
        /// @param	offset	An optional parameter for getting an item that is N items newer than the oldest value.
        ///	@return	output	The item from the buffer.

        sample tail(int offset = 0) {

            // calculate the difference between the capacity and our delay so that tail() can be properly offset
            // extra 2 "now" samples to allow for interpolation
            size_t true_offset = m_history.capacity() - integral_size() - 2 + offset;

            return m_interpolator(m_history.tail(true_offset + 2), m_history.tail(true_offset + 1), m_history.tail(true_offset),
                m_history.tail(true_offset - 1), fractional_size());
        }


        /// Write a single sample into the delay.
        ///	@param	new_input	An item to add.

        void write(sample new_input) {
            m_history.write(new_input);
        }


        /// Erase the delay history.

        void clear() {
            m_history.zero();
        }


        /// Change the interpolation algorithm used.
        /// @param	new_type	option from the interpolator::type enum that names algorithm

        void change_interpolation(interpolator::type new_type = interpolator::type::none) {
            m_interpolator.change_interpolation(new_type);
        }


        /// Calculate one sample.
        ///	@return		Calculated sample

        sample operator()(sample x) {
            // write first (then read) so that we can acheive a zero-sample delay
            write(x);
            return tail();
        }


    private:
        circular_storage<sample> m_history;            ///< Memory for storing the delayed samples.
        number                   m_size;               ///< Delay time in samples. May include a fractional component.
        std::size_t              m_size_integral;      ///< The integral component of the delay time.
        double                   m_size_fractional;    ///< The fractional component of the delay time.
        interpolator::proxy<>    m_interpolator{
            interpolator::type::cubic};    ///< The interpolator instance used to produce interpolated output.
    };


}    // namespace c74::min::lib
