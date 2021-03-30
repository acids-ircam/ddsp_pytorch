//// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {


    /// Limit values to within a specified range, clamping the values to the outer bounds of the range if neccessary.
    ///	@param	input		The value to constrain.
    ///	@param	low_bound	The low bound for the range.
    ///	@param	high_bound	The high bound for the range.
    ///	@return				Returns the value a constrained to the range specified by low_bound and high_bound.
    /// @see				clamp()
    /// @see				wrap()
    /// @see				fold()

#ifdef WIN_VERSION
    #define MIN_CLAMP(input, low_bound, high_bound) clamp<std::remove_reference<decltype(input)>::type>(input, (decltype(input))low_bound, (decltype(input))high_bound)
#else
    #define MIN_CLAMP(input, low_bound, high_bound) clamp<__typeof(input)>(input, low_bound, high_bound)
#endif


    ///	Determine if a value is a power-of-two. Only works for ints.
    ///	@tparam	T		The type of integer to use for the determination.
    ///	@param	value	The value to test.
    ///	@return			True if the input is a power of two, otherwise false.
    ///	@see			limit_to_power_of_two()

    template<class T>
    bool is_power_of_two(const T value) {
        // TODO: static_assert is_integral
        return (value > 0) && ((value & (value - 1)) == 0);
    }


    ///	Limit input to power-of-two values.
    /// Non-power-of-two values are increased to the next-highest power-of-two upon return.
    /// Only works for ints up to 32-bits.
    ///	@tparam	T		The type of integer to use for the determination.
    ///	@param	value	The value to test.
    ///	@see			is_power_of_two
    /// @seealso		 http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2

    template<class T>
    T limit_to_power_of_two(const T a_value) {
        // TODO: static_assert correct type

        T value {a_value};

        value--;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        ++value;
        return value;
    }


    using std::clamp;


    /// Limit values to within a specified range, wrapping the values to within the range if neccessary.
    /// This routine wraps around the range as many times as is needed to get the values in range.
    /// @tparam	T			The data type of the number to be constrained.
    ///	@param	input		The value to constrain.
    ///	@param	low_bound	The low bound for the range.
    ///	@param	high_bound	The high bound for the range.
    ///	@return				Returns the value a wrapped into the range specified by low_bound and high_bound.
    /// @see				c74::max::clamp()
    /// @see				wrap_once()
    /// @see				fold()

    template<class T>
    T wrap(const T input, const T a_low_bound, const T a_high_bound) {
		T low_bound {a_low_bound};
		T high_bound {a_high_bound};

        if (low_bound > high_bound)
            std::swap(low_bound, high_bound);

        double x     = input - low_bound;
        auto   range = high_bound - low_bound;

        if (range == 0)
            return 0;    // don't divide by zero

        if (x > range) {
            if (x > range * 2.0) {
                double d  = x / range;
                long   di = static_cast<long>(d);
                d         = d - di;
                x         = d * range;
            }
            else {
                x -= range;
            }
        }
        else if (x < 0.0) {
            if (x < -range) {
                double d  = x / range;
                long   di = static_cast<long>(d);
                d         = d - di;
                x         = d * range;
            }
            x += range;
        }

        auto result = x + low_bound;
        if (result >= high_bound)
            result -= range;
        return static_cast<T>(result);
    }


    /// A fast routine for wrapping around the range once.
    /// This is faster than doing an expensive modulo, where you know the range of the input
    /// will not equal or exceed twice the range.
    /// @tparam	T			The data type of the number to be constrained.
    ///	@param	input		The value to constrain.
    ///	@param	low_bound	The low bound for the range.
    ///	@param	high_bound	The high bound for the range.
    ///	@return				Returns the value a wrapped into the range specified by low_bound and high_bound.
    /// @see				c74::max::clamp()
    /// @see				wrap()
    /// @see				fold()

    template<class T>
    T wrap_once(const T input, const T low_bound, const T high_bound) {
        if ((input >= low_bound) && (input < high_bound))
            return input;
        else if (input >= high_bound)
            return ((low_bound - 1) + (input - high_bound));
        else
            return ((high_bound + 1) - (low_bound - input));
    }


    /// This routine folds numbers into the data range
    /// Limit values to within a specified range, folding the values to within the range if neccessary.
    /// This routine folds around the range as many times as is needed to get the values in range.
    /// @tparam	T			The data type of the number to be constrained.
    ///	@param	input		The value to constrain.
    ///	@param	low_bound	The low bound for the range.
    ///	@param	high_bound	The high bound for the range.
    ///	@return				Returns the value a folded into the range specified by low_bound and high_bound.
    /// @see				c74::max::clamp()
    /// @see				wrap()

    template<typename T>
    T fold(const T input, T low_bound, T high_bound) {
        if (low_bound > high_bound)
            std::swap(low_bound, high_bound);

        if ((input >= low_bound) && (input <= high_bound))
            return input;    // nothing to fold
        else {
            double fold_range = 2.0 * fabs(static_cast<double>(low_bound - high_bound));
            return fabs(remainder(input - low_bound, fold_range)) + low_bound;
        }
    }


    /// A utility for scaling one range of values onto another range of values.
    /// @tparam	T			The data type of the number to be constrained.
    ///	@param	value		The value to constrain.
    ///	@param	in_low		The low bound for the range of the input.
    ///	@param	in_high		The high bound for the range of the input.
    ///	@param	out_low		The low bound for the range of the output.
    ///	@param	out_high	The high bound for the range of the output.
    ///	@return				Returns the scaled value.

    template<class T>
    static T scale(const T value, const T in_low, const T in_high, const T out_low, const T out_high) {
        const auto in_diff = static_cast<number>(in_high - in_low);
        const auto in_scale = (in_diff != 0.0) ? (1.0 / in_diff) : 1.0;
        const auto out_diff = out_high - out_low;
        const auto normalized = static_cast<T>((value - in_low) * in_scale);
        return static_cast<T>((normalized * out_diff) + out_low);
    }


    /// A utility for scaling one range of values onto another range of values with an exponential curve.
    /// @tparam	T			The data type of the number to be constrained.
    ///	@param	value		The value to constrain.
    ///	@param	in_low		The low bound for the range of the input.
    ///	@param	in_high		The high bound for the range of the input.
    ///	@param	out_low		The low bound for the range of the output.
    ///	@param	out_high	The high bound for the range of the output.
    ///	@param	power		An exponent to be applied in the scaling.
    ///						This argument must be a greater than 0.
    ///						A value of 1.0 produces linear scaling,
    ///						higher values result in an exponential mapping and lower values of result in a logarithmic
    /// scaling.
    ///	@return				Returns the scaled value.

    template<class T>
    static T scale(const T value, const T in_low, const T in_high, const T out_low, const T out_high, const number power) {
        // TODO: ensure that power is > 0.0
        number in_scale = 1 / (in_high - in_low);
        number out_diff = out_high - out_low;
        T out = (value - in_low) * in_scale;
        if (out > 0.0)
            out = pow(out, power);
        else if (out < 0.0)
            out = -pow(-out, power);
        out = (out * out_diff) + out_low;
        return out;
    }


    /// Classes defined in the #limit namespace wrap available range-constraining functions
    /// such that they are suitable for use in specializing other classes.
    /// Most notably these range-constraining functions are used to specialize the min::attribute<> class.
    ///
    /// Exercise caution when using the functions defined here with unsigned values.
    /// Negative, signed integers have the potential to become very large numbers when casting to unsigned integers.
    /// This can cause errors during a boundary check, such as values clipping to the high boundary instead of the
    /// low boundary or numerous iterations of loop to bring a wrapped value back into the acceptable range.

    namespace limit {

        /// The interface for all attribute range limiter classes
        /// @tparam	T The numerical type to be constrained.

        template<typename T>
        class base {
        public:
            /// Constrain input values to the specified range.
            /// @param	input	The input value to constrain.
            /// @param	low		The low boundary of the range.
            /// @param	high	The high boundary of the range.
            /// @return			The constrained value.

            virtual T operator()(const T input, const T low, const T high) = 0;
        };


        /// Attribute range limiter that does not constrain values.
        /// @tparam T	The numerical type to be constrained.

        template<typename T>
        class none : public base<T> {
        public:
            /// Constrain input values to the specified range.
            /// @param	input	The input value to constrain.
            /// @param	low		The low boundary of the range.
            /// @param	high	The high boundary of the range.
            /// @return			The constrained value.

            static T apply(const T input, const T low, const T high) {
                return input;
            }


            /// Constrain input values to the specified range.
            /// @param	input	The input value to constrain.
            /// @param	low		The low boundary of the range.
            /// @param	high	The high boundary of the range.
            /// @return			The constrained value.

            T operator()(const T input, const T low, const T high) {
                return apply(input, low, high);
            }
        };


        /// Attribute range limiter that constrains values by clamping.
        /// @tparam	T The numerical type to be constrained.

        template<typename T>
        class clamp : public base<T> {
        public:
            /// Constrain input values to the specified range.
            /// @param	input	The input value to constrain.
            /// @param	low		The low boundary of the range.
            /// @param	high	The high boundary of the range.
            /// @return			The constrained value.

            static T apply(const T input, const T low, const T high) {
                return min::clamp<T>(input, low, high);
            }


            /// Constrain input values to the specified range.
            /// @param	input	The input value to constrain.
            /// @param	low		The low boundary of the range.
            /// @param	high	The high boundary of the range.
            /// @return			The constrained value.

            T operator()(const T input, const T low, const T high) {
                return apply(input, low, high);
            }
        };


        /// Attribute range limiter that constrains values by wrapping.
        /// @tparam	T The numerical type to be constrained.

        template<typename T>
        class wrap : public base<T> {
        public:
            /// Constrain input values to the specified range.
            /// @param	input	The input value to constrain.
            /// @param	low		The low boundary of the range.
            /// @param	high	The high boundary of the range.
            /// @return			The constrained value.

            static T apply(const T input, const T low, const T high) {
                return min::wrap(input, low, high);
            }


            /// Constrain input values to the specified range.
            /// @param	input	The input value to constrain.
            /// @param	low		The low boundary of the range.
            /// @param	high	The high boundary of the range.
            /// @return			The constrained value.

            T operator()(const T input, const T low, const T high) {
                return apply(input, low, high);
            }
        };


        /// Attribute range limiter that constrains values by folding.
        /// @tparam	T The numerical type to be constrained.

        template<typename T>
        class fold : public base<T> {
        public:
            /// Constrain input values to the specified range.
            /// @param	input	The input value to constrain.
            /// @param	low		The low boundary of the range.
            /// @param	high	The high boundary of the range.
            /// @return			The constrained value.

            static T apply(const T input, const T low, const T high) {
                return min::fold(input, low, high);
            }


            /// Constrain input values to the specified range.
            /// @param	input	The input value to constrain.
            /// @param	low		The low boundary of the range.
            /// @param	high	The high boundary of the range.
            /// @return			The constrained value.

            T operator()(const T input, const T low, const T high) {
                return apply(input, low, high);
            }
        };

    }    // namespace limit


}    // namespace c74::min
