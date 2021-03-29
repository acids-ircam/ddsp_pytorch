/// @file
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

// Visual Studio 2015 doesn't have full support for constexpr
#if !defined(_MSC_VER) || (_MSC_VER > 1900)
#define MIN_CONSTEXPR constexpr
#else
#define MIN_CONSTEXPR
#endif

namespace c74::min::lib {

    /// Defines several methods for <a href="http://en.wikipedia.org/wiki/Interpolation">interpolating</a> between discrete data points such
    /// as those found in an array or matrix. These methods are commonly used in digital audio whenever we alter the rate at which a signal
    /// is read. These functions require known discrete values to be passed by reference along with a double between 0 and 1 representing
    /// the fractional location desired. They return the interpolated value.

    namespace interpolator {


        ///	Shared base class for all interpolator types.

        template<class T = number>
        class base {
        protected:
            MIN_CONSTEXPR base() noexcept {}

        public:
            virtual ~base() {}

            virtual T operator()(T x1, T x2, double delta) noexcept {
                return x1;
            }
            virtual T operator()(T x0, T x1, T x2, T x3, double delta) noexcept {
                return x1;
            }
            virtual void bias(double new_bias) {
                ;
            }
            virtual double bias() {
                return 0.0;
            }
            virtual void tension(double new_tension) {
                ;
            }
            virtual double tension() {
                return 0.0;
            }
        };


        ///	No interpolation always returns the first sample passed to it.
        ///	@tparam	T		The data type to interpolate. By default this is the number type.

        template<class T = number>
        class none : public base<T> {
        public:
            static const int delay = 0;

            /// Interpolate based on 2 samples of input.
            /// @param x1		Sample value that will be returned
            /// @param x2		Unused sample value
            /// @param delta	Unused fractional location
            /// @return         The interpolated value

            MIN_CONSTEXPR T operator()(T x1, T x2, double delta) noexcept {
                return x1;
            }


            /// Interpolate based on 4 samples of input.
            /// @param x0		Unused sample value
            /// @param x1		Sample value that will be returned
            /// @param x2		Unused sample value
            /// @param x3		Unused sample value
            /// @param delta	Unused fractional location
            /// @return         The interpolated value

            MIN_CONSTEXPR T operator()(T x0, T x1, T x2, T x3, double delta) noexcept {
                return x1;
            }
        };


        /// Nearest interpolator returns the closest sample by rounding the delta up or down.
        ///	@tparam	T		The data type to interpolate. By default this is the number type.

        template<class T = number>
        class nearest : public base<T> {
        public:
            static const int delay = 0;

            /// Interpolate based on 2 samples of input.
            /// @param x1		Returned sample value when rounding down
            /// @param x2		Returned sample value when rounding up
            /// @param delta	Fractional location between x1 and x2 @n
            ///                 delta < 0.5 => x1 @n
            ///                 delta >= 0.5 => x2
            /// @return         The interpolated value

            MIN_CONSTEXPR T operator()(T x1, T x2, double delta) noexcept {
                T out = delta < 0.5 ? x1 : x2;
                return out;
            }


            /// Interpolate based on 4 samples of input.
            /// @param x0		Unused sample value
            /// @param x1		Returned sample value when rounding down
            /// @param x2		Returned sample value when rounding up
            /// @param x3		Unused sample value
            /// @param delta	Fractional location between x1 and x2 @n
            ///                 delta < 0.5 => x1 @n
            ///                 delta >= 0.5 => x2
            /// @return         The interpolated value

            MIN_CONSTEXPR T operator()(T x0, T x1, T x2, T x3, double delta) noexcept {
                T out = delta < 0.5 ? x1 : x2;
                return out;
            }
        };

        /// Linear interpolator.
        ///	@tparam	T		The data type to interpolate. By default this is the number type.

        template<class T = number>
        class linear : public base<T> {
        public:
            static const int delay = 1;

            /// Interpolate based on 2 samples of input.
            /// @param x1		Sample value at prior integer index
            /// @param x2		Sample value at next integer index
            /// @param delta 	Fractional location between x1 (delta=0) and x2 (delta=1)
            /// @return			The interpolated value

            MIN_CONSTEXPR T operator()(T x1, T x2, double delta) noexcept {
                return x1 + delta * (x2 - x1);
            }


            /// Interpolate based on 4 samples of input.
            /// @param x0		Unused sample value
            /// @param x1		Sample value at prior integer index
            /// @param x2		Sample value at next integer index
            /// @param x3		Unused sample value
            /// @param delta 	Fractional location between x1 (delta=0) and x2 (delta=1)
            /// @return			The interpolated value

            MIN_CONSTEXPR T operator()(T x0, T x1, T x2, T x3, double delta) noexcept {
                return (*this)(x1, x2, delta);
            }
        };


        /// Allpass interpolator
        /// Testing shows this algorithm will become less accurate the more points it computes between two known samples.
        /// Also, because it uses an internal history, the reset() function should be used when switching between non-continuous segments of
        /// sampled audio data.
        ///	@tparam	T		The data type to interpolate. By default this is the number type.

        template<class T = number>
        class allpass : public base<T> {
        public:
            static const int delay = 1;

            /// Interpolate based on 2 samples of input.
            /// @param x1		Sample value at prior integer index
            /// @param x2		Sample value at next integer index
            /// @param delta 	Fractional location between x1 (delta=0) and x2 (delta=1) @n
            ///            		Be aware that delta=1.0 may not return the exact value at x2 given the nature of this algorithm.
            /// @return			The interpolated value

            MIN_CONSTEXPR T operator()(T x1, T x2, double delta) noexcept {
                T out = x1 + delta * (x2 - mY1);
                mY1   = out;
                return out;
            }


            /// Interpolate based on 4 samples of input.
            /// @param x0		Unused sample value
            /// @param x1		Sample value at prior integer index
            /// @param x2		Sample value at next integer index
            /// @param x3		Unused sample value
            /// @param delta 	Fractional location between x1 (delta=0) and x2 (delta=1) @n
            ///            		Be aware that delta=1.0 may not return the exact value at x2 given the nature of this algorithm.
            /// @return			The interpolated value

            MIN_CONSTEXPR T operator()(T x0, T x1, T x2, T x3, double delta) noexcept {
                return (*this)(x1, x2, delta);
            }


            /// Reset the interpolator history.
            /// This interpolator accumulates a history due to being an IIR filter internally.

            void reset() {
                mY1 = T(0.0);
            }

        private:
            T mY1 = T(0.0);
        };


        /// Cosine interpolator
        ///	@tparam	T		The data type to interpolate. By default this is the number type.

        template<class T = number>
        class cosine : public base<T> {
        public:
            static const int delay = 1;

            /// Interpolate based on 2 samples of input.
            /// @param x1		Sample value at prior integer index
            /// @param x2		Sample value at next integer index
            /// @param delta 	Fractional location between x1 (delta=0) and x2 (delta=1)
            /// @return			The interpolated value

            MIN_CONSTEXPR T operator()(T x1, T x2, double delta) noexcept {
                T a = 0.5 * (1.0 - cos(delta * M_PI));
                return x1 + a * (x2 - x1);
            }


            /// Interpolate based on 4 samples of input.
            /// @param x0		Unused sample value
            /// @param x1		Sample value at prior integer index
            /// @param x2		Sample value at next integer index
            /// @param x3		Unused sample value
            /// @param delta 	Fractional location between x1 (delta=0) and x2 (delta=1)
            /// @return			The interpolated value

            MIN_CONSTEXPR T operator()(T x0, T x1, T x2, T x3, double delta) noexcept {
                return (*this)(x1, x2, delta);
            }
        };


        /// Cubic interpolator
        ///	@tparam	T		The data type to interpolate. By default this is the number type.

        template<class T = number>
        class cubic : public base<T> {
        public:
            static const int delay = 3;

            /// Interpolate based on 4 samples of input.
            /// @param x0		Sample value at integer index prior to x0
            /// @param x1		Sample value at prior integer index
            /// @param x2		Sample value at next integer index
            /// @param x3		Sample value at integer index after x2
            /// @param delta	Fractional location between x1 (delta=0) and x2 (delta=1)
            /// @return			The interpolated value

            MIN_CONSTEXPR T operator()(T x0, T x1, T x2, T x3, double delta) noexcept {
                double delta2 = delta * delta;
                T      a      = x3 - x2 - x0 + x1;
                T      b      = x0 - x1 - a;
                T      c      = x2 - x0;
                return a * delta * delta2 + b * delta2 + c * delta + x1;
            }
        };


        /// Spline interpolator based on the Breeuwsma catmull-rom spline
        ///	@tparam	T		The data type to interpolate. By default this is the number type.

        template<class T = number>
        class spline : public base<T> {
        public:
            static const int delay = 3;

            /// Interpolate based on 4 samples of input.
            /// @param x0	Sample value at integer index prior to x
            /// @param x1	Sample value at prior integer index
            /// @param x2	Sample value at next integer index
            /// @param x3	Sample value at integer index after y
            /// @param delta	Fractional location between x1 (delta=0) and x2 (delta=1)
            /// @return		The interpolated value.

            MIN_CONSTEXPR T operator()(T x0, T x1, T x2, T x3, double delta) noexcept {
                T delta2 = delta * delta;
                T f0     = -0.5 * x0 + 1.5 * x1 - 1.5 * x2 + 0.5 * x3;
                T f1     = x0 - 2.5 * x1 + 2.0 * x2 - 0.5 * x3;
                T f2     = -0.5 * x0 + 0.5 * x2;
                return f0 * delta * delta2 + f1 * delta2 + f2 * delta + x1;
            }
        };


        /// Hermite interpolator
        /// When bias and tension are both set to 0.0, this algorithm is equivalent to Spline.
        ///	@tparam	T		The data type to interpolate. By default this is the number type.

        template<class T = number>
        class hermite : public base<T> {
        public:
            static const int delay{3};

            /// Set the bias attribute.
            /// @param	new_bias	The new bias value used in interpolating.

            void bias(double new_bias) {
                m_bias = new_bias;
            }


            /// Return the value of the bias attribute
            /// @return The current bias.

            double bias() {
                return m_bias;
            }


            /// Set the tension attribute.
            /// @param	new_tension		The new tension value used in interpolating.

            void tension(double new_tension) {
                m_tension = new_tension;
            }


            /// Return the value of the tension attribute
            /// @return The current tension.

            double tension() {
                return m_tension;
            }


            /// Interpolate based on 4 samples of input.
            /// @param x0	Sample value at integer index prior to x
            /// @param x1	Sample value at prior integer index
            /// @param x2	Sample value at next integer index
            /// @param x3	Sample value at integer index after y
            /// @param delta	Fractional location between x1 (delta=0) and x2 (delta=1)
            /// @return		The interpolated value.

            MIN_CONSTEXPR T operator()(T x0, T x1, T x2, T x3, double delta) noexcept {
                T delta2 = delta * delta;
                T delta3 = delta * delta2;
                T bp     = 1 + m_bias;
                T bm     = 1 - m_bias;
                T mt     = (1 - m_tension) * 0.5;
                T m0     = ((x1 - x0) * bp + (x2 - x1) * bm) * mt;
                T m1     = ((x2 - x1) * bp + (x3 - x2) * bm) * mt;
                T a0     = 2 * delta3 - 3 * delta2 + 1;
                T a1     = delta3 - 2 * delta2 + delta;
                T a2     = delta3 - delta2;
                T a3     = -2 * delta3 + 3 * delta2;
                return a0 * x1 + a1 * m0 + a2 * m1 + a3 * x2;
            }

        private:
            double m_bias{0.0};       // attribute
            double m_tension{0.0};    // attribute
        };


        /// Contains the names of available interpolation algorithms.
        /// Used with proxy::change_interpolation() to select a specific option.
        /// The final definition "type_count" is included to provide a method for querying the size of this enum list.

        enum class type : unsigned int { none, nearest, linear, allpass, cosine, cubic, spline, hermite, type_count };


        /// Proxy that provides means for objects to switch between interpolation types.
        /// @tparam	T		The data type to interpolate. By default this is the number type.

        template<class T = number>
        class proxy {
        public:
            /// Default constructor
            /// @param	first_type	Option from the type enum. By default this is type::none.

            explicit proxy(interpolator::type first_type = type::none) {
                // NW: The order here must match the order in type enum

                std::get<0>(m_type_vector) = std::unique_ptr<interpolator::base<T>>(new interpolator::none<T>);
                std::get<1>(m_type_vector) = std::unique_ptr<interpolator::base<T>>(new interpolator::nearest<T>);
                std::get<2>(m_type_vector) = std::unique_ptr<interpolator::base<T>>(new interpolator::linear<T>);
                std::get<3>(m_type_vector) = std::unique_ptr<interpolator::base<T>>(new interpolator::allpass<T>);
                std::get<4>(m_type_vector) = std::unique_ptr<interpolator::base<T>>(new interpolator::cosine<T>);
                std::get<5>(m_type_vector) = std::unique_ptr<interpolator::base<T>>(new interpolator::cubic<T>);
                std::get<6>(m_type_vector) = std::unique_ptr<interpolator::base<T>>(new interpolator::spline<T>);
                std::get<7>(m_type_vector) = std::unique_ptr<interpolator::base<T>>(new interpolator::hermite<T>);

                m_which_type = static_cast<int>(first_type);
            }


            /// Set the bias attribute on hermite interpolator.
            /// @param	new_bias	The new bias value used in interpolating.

            void bias(double new_bias) {
                m_type_vector[m_hermite_type]->bias(new_bias);
            }


            /// Return the value of the bias attribute on hermite interpolator.
            /// @return The current bias.

            double bias() {
                return m_type_vector[m_hermite_type]->bias();
            }


            /// Set the tension attribute on hermite interpolator.
            /// @param	new_tension		The new tension value used in interpolating.

            void tension(double new_tension) {
                m_type_vector[m_hermite_type]->tension(new_tension);
            }


            /// Return the value of the tension attribute on hermite interpolator.
            /// @return The current tension.

            double tension() {
                return m_type_vector[m_hermite_type]->tension();
            }


            /// Interpolate based on 4 samples of input.
            /// @param x0		Unused sample value
            /// @param x1		Sample value that will be returned
            /// @param x2		Unused sample value
            /// @param x3		Unused sample value
            /// @param delta	Unused fractional locationq
            /// @return         The interpolated value

            MIN_CONSTEXPR T operator()(T x0, T x1, T x2, T x3, double delta) noexcept {
                return m_type_vector[m_which_type]->operator()(x0, x1, x2, x3, delta);
            }


            /// Change the interpolation algorithm used.
            /// @param	new_type	option from the type enum

            void change_interpolation(type new_type) {
                m_which_type = static_cast<int>(new_type);
            }


        private:
            std::array<std::unique_ptr<interpolator::base<T>>, std::size_t(type::type_count)> m_type_vector;    ///< vector with one instance of each interpolator type
            int m_which_type;    ///< index within m_type_vector used for interpolation operator, stored to avoid repeat casting
            static const int m_hermite_type
                = static_cast<int>(type::hermite);    ///< index of the hermite interpolator, stored to avoid repeat casting
        };

    }    // namespace interpolator
}      // namespace c74::min::lib
