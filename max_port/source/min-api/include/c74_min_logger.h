/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {

    class logger_line_ending {};       ///< A type to represent line endings for the min::logger class.
    static logger_line_ending endl;    ///< An instance of a line ending for convenience.


    /// Logging utility to deliver console messages
    /// in such a way that they are mirrored to the Max window.
    ///
    /// This class is not intended to be used directly,
    /// but rather through instances that are provided in min::object<> base class.
    ///
    /// @see min::object::cout
    /// @see min::object::cerr
    /// @see min::endl

    class logger {
    public:
        /// The output type of the message.
        /// These are not `levels` as in some languages (e.g. Ruby) but distinct targets.

        enum class type {
            message = 0,    ///< A regular console post to the Max Window.
            warning,		///< A highlighted and trappable warning post to the Max Window.
            error           ///< A highlighted and trappable error post to the Max Window.
        };


        /// Constructor: typically you do not call this directly,
        /// it used by min::object to create cout and cerr
        /// @param an_owner		Your object instance
        /// @param a_type		The type of console output to deliver

        logger(const object_base* an_owner, const logger::type a_type)
        : m_owner { *an_owner }
        , m_target { a_type }
        {}


        /// Use the insertion operator as for any other stream to build the output message
        /// @param	x	A token to be added to the output stream.
        /// @return		A reference to the output stream.

        template<typename T>
        logger& operator<<(const T& x) {
            m_stream << x;
            return *this;
        }


        /// Pass endl to the insertion operator to complete the console post and flush it.
        /// @param x	The min::endl token
        /// @return		A reference to the output stream.

        logger& operator<<(const logger_line_ending& x) {
            const std::string& s = m_stream.str();

            switch (m_target) {
				case type::message:
                    std::cout << s << std::endl;

                    // if the max object is present then it is safe to post even if the owner isn't yet fully initialized
                    if (m_owner.initialized() || k_sym_max)
                        max::object_post(m_owner, s.c_str());
                    break;
				case type::warning:
                    std::cerr << s << std::endl;

                    // if the max object is present then it is safe to post even if the owner isn't yet fully initialized
                    if (m_owner.initialized() || k_sym_max)
                        max::object_warn(m_owner, s.c_str());
                    break;
				case type::error:
                    std::cerr << s << std::endl;

                    // if the max object is present then it is safe to post even if the owner isn't yet fully initialized
                    if (m_owner.initialized() || k_sym_max)
                        max::object_error(m_owner, s.c_str());
                    break;
            }

            m_stream.str("");
            return *this;
        }

    private:
        const object_base&  m_owner;
        const logger::type  m_target;
        std::stringstream   m_stream;
    };

}    // namespace c74::min
