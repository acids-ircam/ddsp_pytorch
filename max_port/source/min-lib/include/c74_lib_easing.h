/// @file
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min::lib::easing {


    /// The "linear" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T linear(T x) {
        return x;
    }


    /// The "in-back" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_back(T x) {
        return x * x * x - x * sin(x * M_PI);
    }


    /// The "in-out-back" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_out_back(T x) {
        if (x < 0.5) {
            double f = 2 * x;
            return 0.5 * (f * f * f - f * sin(f * M_PI));
        }
        else {
            double f = (1 - (2 * x - 1));
            return 0.5 * (1 - (f * f * f - f * sin(f * M_PI))) + 0.5;
        }
    }


    /// The "out-back" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T out_back(T x) {
        double f = 1.0 - x;
        return 1 - (f * f * f - f * sin(f * M_PI));
    }


    /// The "in-bounce" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_bounce(T x) {
        double f = 1.0 - x;
        double y;

        if (f < 4 / 11.0)
            y = (121 * f * f) / 16.0;
        else if (f < 8 / 11.0)
            y = (363 / 40.0 * f * f) - (99 / 10.0 * f) + 17 / 5.0;
        else if (f < 9 / 10.0)
            y = (4356 / 361.0 * f * f) - (35442 / 1805.0 * f) + 16061 / 1805.0;
        else
            y = (54 / 5.0 * f * f) - (513 / 25.0 * f) + 268 / 25.0;
        return 1.0 - y;
    }


    /// The "in-out-bounce" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_out_bounce(T x) {
        if (x < 0.5) {
            double f = 1 - 2 * x;
            double y;

            if (f < 4 / 11.0)
                y = (121 * f * f) / 16.0;
            else if (f < 8 / 11.0)
                y = (363 / 40.0 * f * f) - (99 / 10.0 * f) + 17 / 5.0;
            else if (f < 9 / 10.0)
                y = (4356 / 361.0 * f * f) - (35442 / 1805.0 * f) + 16061 / 1805.0;
            else
                y = (54 / 5.0 * f * f) - (513 / 25.0 * f) + 268 / 25.0;
            return 0.5 * (1 - y);
        }
        else {
            double f = x * 2 - 1;
            double y;

            if (f < 4 / 11.0)
                y = (121 * f * f) / 16.0;
            else if (f < 8 / 11.0)
                y = (363 / 40.0 * f * f) - (99 / 10.0 * f) + 17 / 5.0;
            else if (f < 9 / 10.0)
                y = (4356 / 361.0 * f * f) - (35442 / 1805.0 * f) + 16061 / 1805.0;
            else
                y = (54 / 5.0 * f * f) - (513 / 25.0 * f) + 268 / 25.0;
            return 0.5 * y + 0.5;
        }
    }


    /// The "out-bounce" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T out_bounce(T x) {
        if (x < 4 / 11.0)
            return (121 * x * x) / 16.0;
        else if (x < 8 / 11.0)
            return (363 / 40.0 * x * x) - (99 / 10.0 * x) + 17 / 5.0;
        else if (x < 9 / 10.0)
            return (4356 / 361.0 * x * x) - (35442 / 1805.0 * x) + 16061 / 1805.0;
        else
            return (54 / 5.0 * x * x) - (513 / 25.0 * x) + 268 / 25.0;
    }


    /// The "in-circular" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_circular(T x) {
        x = 1 - x * x;
        return 1 - sqrt( std::max(x, 0.0) );
    }


    /// The "in-out-circular" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_out_circular(T x) {
        if (x < 0.5) {
            x = 1 - 4 * (x * x);
            return 0.5 * (1 - sqrt( std::max(x, 0.0) ));
        }
        else {
            x = -((2 * x) - 3) * ((2 * x) - 1);
            return 0.5 * (sqrt( std::max(x, 0.0) ) + 1);
        }
    }


    /// The "out-circular" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T out_circular(T x) {
        x = (2 - x) * x;
        return sqrt( std::max(x, 0.0) );
    }


    /// The "in-cubic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_cubic(T x) {
        return x * x * x;
    }


    /// The "in-out-cubic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_out_cubic(T x) {
        if (x < 0.5)
            return 4.0 * x * x * x;
        else {
            double f = ((2 * x) - 2);
            return 0.5 * f * f * f + 1;
        }
    }


    /// The "out-cubic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T out_cubic(T x) {
        double f = x - 1.0;
        return f * f * f + 1.0;
    }


    /// The "in-elastic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_elastic(T x) {
        return sin(6.5 * M_PI * x) * pow(2, 10 * (x - 1));
    }


    /// The "in-out-elastic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_out_elastic(T x) {
        if (x < 0.5)
            return 0.5 * sin(6.5 * M_PI * (2 * x)) * pow(2, 10 * ((2 * x) - 1));
        else
            return 0.5 * (sin(-6.5 * M_PI * ((2 * x - 1) + 1)) * pow(2, -10 * (2 * x - 1)) + 2);
    }


    /// The "out-elastic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T out_elastic(T x) {
        return sin(-6.5 * M_PI * (x + 1)) * pow(2, -10 * x) + 1;
    }


    /// The "in-exponential" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_exponential(T x) {
        if (x == 0.0)
            return x;
        else
            return pow(2, 10 * (x - 1));
    }


    /// The "in-out-exponential" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_out_exponential(T x) {
        if (x == 0.0 || x == 1.0)
            return x;
        else if (x < 0.5)
            return 0.5 * pow(2, (20 * x) - 10);
        else
            return -0.5 * pow(2, (-20 * x) + 10) + 1;
    }


    /// The "out-exponential" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T out_exponential(T x) {
        if (x == 1.0)
            return x;
        else
            return 1 - pow(2, -10 * x);
    }


    /// The "in-quadratic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_quadratic(T x) {
        return x * x;
    }


    /// The "in-out-quadratic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_out_quadratic(T x) {
        if (x < 0.5)
            return 2 * x * x;
        else
            return (-2 * x * x) + (4 * x) - 1;
    }


    /// The "out-quadratic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T out_quadratic(T x) {
        return -(x * (x - 2));
    }


    /// The "in-quartic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_quartic(T x) {
        return x * x * x * x;
    }


    /// The "in-out-quartic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_out_quartic(T x) {
        if (x < 0.5)
            return 8 * x * x * x * x;
        else {
            double f = (x - 1);
            return -8 * f * f * f * f + 1;
        }
    }


    /// The "out-quartic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T out_quartic(T x) {
        double f = (x - 1);
        return f * f * f * (1 - x) + 1;
    }


    /// The "in-quintic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_quintic(T x) {
        return x * x * x * x * x;
    }


    /// The "in-out-quintic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_out_quintic(T x) {
        if (x < 0.5)
            return 16 * x * x * x * x * x;
        else {
            double f = ((2 * x) - 2);
            return 0.5 * f * f * f * f * f + 1;
        }
    }


    /// The "out-quintic" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T out_quintic(T x) {
        double f = (x - 1);
        return f * f * f * f * f + 1;
    }


    /// The "in-sine" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_sine(T x) {
        return sin((x - 1) * M_PI * 0.5) + 1;
    }


    /// The "in-out-sine" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T in_out_sine(T x) {
        return 0.5 * (1 - cos(x * M_PI));
    }


    /// The "out-sine" easing function as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The output of the easing function.

    template<typename T>
    T out_sine(T x) {
        return sin(x * M_PI * 0.5);
    }


    /// An enum to represent a selection any of the available easing functions in an object attribute.
    /// @see easing::apply()
    /// @see easing::function_info
    /// @see attribute

    enum class function {
        linear,

        in_back,           ///< in-back
        in_bounce,         ///< in-bounce
        in_circular,       ///< in-circular
        in_cubic,          ///< in-cubic
        in_elastic,        ///< in-elastic
        in_exponential,    ///< in-exponential
        in_quadratic,      ///< in-quadratic
        in_quartic,        ///< in-quartic
        in_quintic,        ///< in-quintic
        in_sine,           ///< in-sine

        in_out_back,           ///< in-out-back
        in_out_bounce,         ///< in-out-bounce
        in_out_circular,       ///< in-out-circular
        in_out_cubic,          ///< in-out-cubic
        in_out_elastic,        ///< in-out-elastic
        in_out_exponential,    ///< in-out-exponential
        in_out_quadratic,      ///< in-out-quadratic
        in_out_quartic,        ///< in-out-quartic
        in_out_quintic,        ///< in-out-quintic
        in_out_sine,           ///< in-out-sine

        out_back,           ///< out-back
        out_bounce,         ///< out-bounce
        out_circular,       ///< out-circular
        out_cubic,          ///< out-cubic
        out_elastic,        ///< out-elastic
        out_exponential,    ///< out-exponential
        out_quadratic,      ///< out-quadratic
        out_quartic,        ///< out-quartic
        out_quintic,        ///< out-quintic
        out_sine,           ///< out-sine

        enum_count    ///< the number of available easing functions
    };


    /// Textual descriptors for the enum values that represent easing functions in an object attribute.
    /// @see easing::function_info
    /// @see attribute

    static enum_map function_info = {"linear",

        "in_back", "in_bounce", "in_circular", "in_cubic", "in_elastic", "in_exponential", "in_quadratic", "in_quartic", "in_quintic",
        "in_sine",

        "in_out_back", "in_out_bounce", "in_out_circular", "in_out_cubic", "in_out_elastic", "in_out_exponential", "in_out_quadratic",
        "in_out_quartic", "in_out_quintic", "in_out_sine",

        "out_back", "out_bounce", "out_circular", "out_cubic", "out_elastic", "out_exponential", "out_quadratic", "out_quartic",
        "out_quintic", "out_sine"};


    /// Apply one of the standard easing functions to a number as formalized and popularized by Robert Penner.
    /// @tparam	T		The type of number to use for the calculations (e.g. float, double, number, or sample).
    /// @param	name	The easing function to apply as enumerated in the #easing::function enum.
    ///	@param	x		The value to feed as input into the easing function.
    ///	@return			The "eased" output.

    template<typename T>
    T apply(easing::function name, T x) {
        switch (name) {
            case easing::function::linear:
                return linear(x);
            case easing::function::in_back:
                return in_back(x);
            case easing::function::in_out_back:
                return in_out_back(x);
            case easing::function::out_back:
                return out_back(x);
            case easing::function::in_bounce:
                return in_bounce(x);
            case easing::function::in_out_bounce:
                return in_out_bounce(x);
            case easing::function::out_bounce:
                return out_bounce(x);
            case easing::function::in_circular:
                return in_circular(x);
            case easing::function::in_out_circular:
                return in_out_circular(x);
            case easing::function::out_circular:
                return out_circular(x);
            case easing::function::in_cubic:
                return in_cubic(x);
            case easing::function::in_out_cubic:
                return in_out_cubic(x);
            case easing::function::out_cubic:
                return out_cubic(x);
            case easing::function::in_elastic:
                return in_elastic(x);
            case easing::function::in_out_elastic:
                return in_out_elastic(x);
            case easing::function::out_elastic:
                return out_elastic(x);
            case easing::function::in_exponential:
                return in_exponential(x);
            case easing::function::in_out_exponential:
                return in_out_exponential(x);
            case easing::function::out_exponential:
                return out_exponential(x);
            case easing::function::in_quadratic:
                return in_quadratic(x);
            case easing::function::in_out_quadratic:
                return in_out_quadratic(x);
            case easing::function::out_quadratic:
                return out_quadratic(x);
            case easing::function::in_quartic:
                return in_quartic(x);
            case easing::function::in_out_quartic:
                return in_out_quartic(x);
            case easing::function::out_quartic:
                return out_quartic(x);
            case easing::function::in_quintic:
                return in_quintic(x);
            case easing::function::in_out_quintic:
                return in_out_quintic(x);
            case easing::function::out_quintic:
                return out_quintic(x);
            case easing::function::in_sine:
                return in_sine(x);
            case easing::function::in_out_sine:
                return in_out_sine(x);
            case easing::function::out_sine:
                return out_sine(x);
            case easing::function::enum_count:
                assert(false);
        }
        return 0.0;
    }


}    // namespace c74::min::lib::easing
