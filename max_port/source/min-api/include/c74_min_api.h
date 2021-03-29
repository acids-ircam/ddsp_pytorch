/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_max.h"
#include "c74_ui.h"
#include "c74_ui_graphics.h"
#include "c74_msp.h"

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <list>
#include <mutex>
#include <queue>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <functional>
#include <unordered_map>
#include <utility>

#ifdef MAC_VERSION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#endif
#include "readerwriterqueue/readerwriterqueue.h"
#ifdef MAC_VERSION
#pragma clang diagnostic pop
#endif

#include "murmur/Murmur3.h"    // used for constexpr hash function

namespace c74::min {

    // types

    using uchar = unsigned char;

    using std::string;
    using std::vector;

    template<class T>
    using unique_ptr = std::unique_ptr<T>;

    using number = double;
    using sample = double;
    struct anything {};

    using numbers = std::vector<number>;
    using ints = std::vector<int>;
    using strings = std::vector<std::string>;

    template<size_t count>
    using samples = std::array<sample, count>;

    using sample_vector = std::vector<sample>;


    // The title and description types are just strings.
    // However, we have to define them unambiguously for the argument parsing in the attribute.

    class title : public std::string {
        using std::string::string;    // inherit constructors
    };

    class description : public std::string {
        using std::string::string;    // inherit constructors
    };


    /// This enum thus represents a placeholder type that is used in places where dummy template parameters are present.

    enum class placeholder {
        none    ///< No flags, functions, or other alterations to the class.
    };


    enum class message_type : long {
        no_argument,
        int_argument,
        float_argument,
        symbol_argument,
        object_argument,
        int_optional,
        float_optional,
        symbol_optional,
        gimme,
        cant,
        semicolon,
        comma,
        dollar,
        dollar_symbol,
        gimmeback,
        defer     = max::A_DEFER,
        usurp     = max::A_USURP,
        defer_low = max::A_DEFER_LOW,
        usurp_low = max::A_USURP_LOW,
        ellipsis
    };


    // Very selective group from the STL used only for making common
    // template SFINAE code more readable

    using std::enable_if;
    using std::is_base_of;
    using std::is_enum;
    using std::is_same;


    // Helper code for type/template selection

    class symbol;
    class time_value;
    class matrix_operator_base;
    class gl_operator_base;
    class mc_operator_base;
    class sample_operator_base;
    class vector_operator_base;
    class ui_operator_base;

    namespace ui {
        class color {
        public:
            enum class predefined { black, white, gray };

            color() {}

            color(const max::t_jrgba a_color)
            : m_color {a_color} {}

            color(const double red, const double green, const double blue, const double alpha)
            : m_color{red, green, blue, alpha} {}

            color(const predefined a_color) {
                switch (a_color) {
                    case predefined::black:
                        m_color = {0.0, 0.0, 0.0, 1.0};
                        break;
					case predefined::white:
                        m_color = {1.0, 1.0, 1.0, 1.0};
                        break;
					case predefined::gray:
                        m_color = {0.7, 0.7, 0.7, 1.0};
                        break;
                }
            }


            operator c74::max::t_jrgba*() {
                return &m_color;
            }


            double red() const {
                return m_color.red;
            }

            double green() const {
                return m_color.green;
            }

            double blue() const {
                return m_color.blue;
            }

            double alpha() const {
                return m_color.alpha;
            }

            bool operator==(const color& b) const {
                return red() == b.red() && green() == b.green() && blue() == b.blue() && alpha() == b.alpha();
            }

            bool operator!=(const color& b) const {
                return red() != b.red() || green() != b.green() || blue() != b.blue() || alpha() != b.alpha();
            }

        private:
			max::t_jrgba m_color { 0.0, 0.0, 0.0, 1.0 };
        };
    }    // namespace ui


    template<class T>
    using is_class = std::is_class<T>;

    template<class T>
    using is_symbol = is_same<T, symbol>;

    template<class T>
    using is_time_value = is_same<T, time_value>;

    template<class T>
    using is_color = is_same<T, ui::color>;

    template<class min_class_type>
    using enable_if_matrix_operator =
        typename enable_if<is_base_of<matrix_operator_base, min_class_type>::value, int>::type;

    template<class min_class_type>
    using enable_if_not_matrix_operator =
        typename enable_if<!is_base_of<matrix_operator_base, min_class_type>::value, int>::type;

    template<class min_class_type>
    using enable_if_gl_operator =
        typename enable_if<is_base_of<gl_operator_base, min_class_type>::value, int>::type;

    template<class min_class_type>
    using enable_if_mc_operator =
        typename enable_if<is_base_of<mc_operator_base, min_class_type>::value, int>::type;

    template<class min_class_type>
    using enable_if_sample_operator =
        typename enable_if<is_base_of<sample_operator_base, min_class_type>::value, int>::type;

    template<class min_class_type>
    using enable_if_vector_operator =
        typename enable_if<is_base_of<vector_operator_base, min_class_type>::value, int>::type;

    template<class min_class_type>
    using enable_if_audio_class =
        typename enable_if<is_base_of<vector_operator_base, min_class_type>::value
        || is_base_of<mc_operator_base, min_class_type>::value
        || is_base_of<sample_operator_base, min_class_type>::value, int>::type;

    template<class min_class_type>
    using enable_if_jitter_class =
        typename enable_if<is_base_of<matrix_operator_base, min_class_type>::value
        || is_base_of<gl_operator_base, min_class_type>::value, int>::type;

    template<class min_class_type>
    using enable_if_not_jitter_class =
        typename enable_if<!is_base_of<matrix_operator_base, min_class_type>::value
        && !is_base_of<gl_operator_base, min_class_type>::value, int>::type;


    template<class min_class_type>
    using type_enable_if_audio_class =
        typename enable_if<is_base_of<vector_operator_base, min_class_type>::value
        || is_base_of<mc_operator_base, min_class_type>::value
        || is_base_of<sample_operator_base, min_class_type>::value>::type;

    template<class min_class_type>
    using type_enable_if_not_audio_class =
        typename enable_if<!is_base_of<vector_operator_base, min_class_type>::value
        && !is_base_of<mc_operator_base, min_class_type>::value
        && !is_base_of<sample_operator_base, min_class_type>::value>::type;

    template<class min_class_type>
    using type_enable_if_not_jitter_class =
        typename enable_if<!is_base_of<matrix_operator_base, min_class_type>::value
        && !is_base_of<gl_operator_base, min_class_type>::value>::type;


    template<class min_class_type>
    using enable_if_ui_operator =
        typename enable_if<is_base_of<ui_operator_base, min_class_type>::value, int>::type;

    template<class min_class_type>
    using enable_if_not_ui_operator =
        typename enable_if<!is_base_of<ui_operator_base, min_class_type>::value, int>::type;

    template<class min_class_type>
    using type_enable_if_not_ui_class =
        typename enable_if<!is_base_of<ui_operator_base, min_class_type>::value>::type;


    enum class threadsafe { undefined, no, yes };
    enum class allow_repetitions { undefined, no, yes };

    using mutex = std::mutex;
    using guard = std::lock_guard<std::mutex>;
    using lock  = std::unique_lock<std::mutex>;


    template<typename T>
    using fifo = moodycamel::ReaderWriterQueue<T>;


    /// Compare two floating-point numbers to determine if they are roughly equal
    /// @param lhs The left hand side of the comparison
    /// @param rhs The right hand side of the comparison
    /// @return true if they are roughly the same, otherwise false

    template<typename T>
    bool equivalent(const T lhs, const T rhs, const double epsilon = std::numeric_limits<float>::epsilon() * 100.0, const double margin = 0.0, const double scale = 1.0) {
        if (std::fabs( lhs - rhs ) < epsilon * (scale + (std::max)( std::fabs(lhs), std::fabs(rhs) ) ))
            return true;
        return std::fabs(lhs - rhs) < margin;
    }

}    // namespace c74::min


namespace c74::min {
    static max::t_class*    this_class                      { nullptr };
    static bool             this_class_init                 { false };
    static max::t_symbol*   this_class_name                 { nullptr };
    static bool             this_class_dummy_constructed    { false };


    /// Find out if the current class instance is a dummy instance.
    /// The dummy instance is used for the initial class reflection and wrapper configuration.
    /// All instances after that point are valid (non-dummy) instances.

    inline bool dummy() {
        return this_class_dummy_constructed == false;
    }


    /// A standard interface for flagging serious runtime snafus.
    /// At the moment this is hardwired to throw an exception but offers us the ability to
    /// change that behavior later or specialize it for certain contexts.
    ///
    /// Because this throws an exception you should **not** call this function in an audio perform routine.

    inline void error(const std::string& description) {
        if (!c74::min::dummy())
            throw std::runtime_error(description);
        else
            std::cerr << description << std::endl;
    }


    /// Throw a generic error.
    /// When possible you should specify a description string and pass as an argument instead of calling this variant.

    inline void error() {
        error("unknown error");
    }


    /// @param	check	The condition which triggers the error.
    ///					In other words "true" will cause an exception to throw while "false" will not.

    inline void error(const bool check, const std::string& description) {
        if (check)
            error(description);
    }


    /// Reverse the byte-ordering of an int.
    /// Meaning from Big Endian to Little or vice versa.
    /// @param x	The int to have its byte-ordering reversed.
    /// @return		The byte-swapped output of this function.

    inline uint16_t byteorder_swap(const uint16_t x) {
        return ((int16_t)(((((uint16_t)(x)) >> 8) & 0x00ff) + ((((uint16_t)(x)) << 8) & 0xff00)));
    }
}

#include "c74_min_string.h"     // String helper functions
#include "c74_min_symbol.h"
#include "c74_min_atom.h"
#include "c74_min_dictionary.h"
#include "c74_min_limit.h"      // Library of miscellaneous helper functions (e.g. range clipping)

#include "c74_min_notification.h"       // A class representing notifications from attached-to objects
#include "c74_min_patcher.h"            // Wrapper for interfacing with patchers

#include "c74_min_object_components.h"  // Shared components of Max objects
#include "c74_jitter.h"
#include "c74_min_flags.h"              // Class flags
#include "c74_min_time.h"               // ITM Support
#include "c74_min_port.h"               // Inlets and Outlets
#include "c74_min_threadsafety.h"       // ...
#include "c74_min_inlet.h"              // ...
#include "c74_min_outlet.h"             // ...
#include "c74_min_argument.h"           // Arguments to objects
#include "c74_min_message.h"            // Messages to objects
#include "c74_min_attribute.h"          // Attributes of objects
#include "c74_min_logger.h"             // Console / Max Window output
#include "c74_min_operator_vector.h"    // Vector-based MSP object add-ins
#include "c74_min_operator_sample.h"    // Sample-based MSP object add-ins
#include "c74_min_operator_mc.h"    	// Vector-based MC object add-ins
#include "c74_min_operator_matrix.h"    // Jitter MOP add-ins
#include "c74_min_operator_ui.h"		// User Interface add-ins
#include "c74_min_graphics.h"			// Graphics classes for UI objects
#include "c74_min_event.h"              // Mouse-event and Touch-event classes

#include "c74_min_object_wrapper.h"     // Max wrapper for Min objects
#include "c74_min_object.h"             // The Min object class that glues it all together

#include "c74_min_timer.h"              // Wrapper for clocks
#include "c74_min_queue.h"              // Wrapper for qelems and fifos
#include "c74_min_buffer.h"             // Wrapper for MSP buffers
#include "c74_min_path.h"               // Wrapper class for accessing the Max path system
#include "c74_min_texteditor.h"         // Wrapper for text editor window
#include "c74_min_dataspace.h"          // Unit conversion routines (e.g. db-to-linear or hz-to-midi)

#include "c74_min_doc.h"                // Instrumentation and tools for generating documentation from Min classes


// Prototype for the actual function that will wrap the Min class as a Max external
// Don't use directly -- use the MIN_EXTERNAL macro instead.

template<class min_class_type, class = void>
void wrap_as_max_external(const char* cppname, const char* maxname, void* resources, min_class_type* instance = nullptr);


/// Wrap a class that extends min::object for use in the Max environment.
/// The name of your Max object will be the same as that of your *source file name* minus the ".cpp" suffix.
/// If your filename ends with "_tilde" the name will be substituted with a "~" character at the end.
/// @param	cpp_classname	The name of your class.
/// @see					MIN_EXTERNAL_CUSTOM

#define MIN_EXTERNAL(cpp_classname)                                                                                                        \
    void ext_main(void* r) {                                                                                                               \
        c74::min::wrap_as_max_external<cpp_classname>(#cpp_classname, __FILE__, r);                                                        \
    }


/// Wrap a class that extends min::object for use in the Max environment.
/// The recommended model is to name your file consistent with Min conventions and
/// use the #MIN_EXTERNAL macro instead of this one.
///
/// @param	cpp_classname	The name of your class.
/// @param	max_name		The name of your object as you will type it into a Max object box.
/// @see					MIN_EXTERNAL

#define MIN_EXTERNAL_CUSTOM(cpp_classname, max_name)                                                                                       \
    void ext_main(void* r) {                                                                                                               \
        c74::min::wrap_as_max_external<cpp_classname>(#cpp_classname, #max_name, r);                                                       \
    }
