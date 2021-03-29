/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {


    /// Flags that determine the behavior of automatically generated documentation (maxref) files.

    enum class documentation_flags : int {
        none,              ///< No flags. Use the default behavior (generate / update files when they are old).
        do_not_generate    ///< Do not generate/update the reference files for this object.
    };


    /// Behavior flags determine how Max will treat your object and make it available within the Max environment.

    enum class behavior_flags : int {
        none,    ///< No flags. Use the default behavior (e.g. allow users to create the object in a box).
        nobox    ///< Cannot create in a max box (i.e. it is an internal-use-only class).
    };


    /// Host flags determine in which environments object use is permitted.

    enum class host_flags : int {
        none,      ///< No flags. Use the default behavior (allow in all hosts: both Max and Live)
        no_live    ///< Do not make this object available in the live host.
    };


    /// An amalgamation that represents all available class-level flags for a Min object.
    /// This class should be created through the use of the #MIN_FLAGS macro in your Min class definition rather than directly.

    class flags {
    public:
        /// Declare flags for your class.
        /// @param	doc		Documentation flags for the class.
        /// @see			documentation_flags

        explicit constexpr flags(documentation_flags doc)
        : m_documentation{doc}
        {}


        /// Declare flags for your class.
        /// @param	behavior	Behavior flags for the class.
        /// @see				behavior_flags

        explicit constexpr flags(behavior_flags behavior)
        : m_behavior{behavior}
        {}


        /// Declare flags for your class.
        /// @param	host	Host flags for the class.
        /// @see			host_flags

        explicit constexpr flags(host_flags host)
        : m_host{host}
        {}


        /// Declare flags for your class.
        /// @param	behavior	Behavior flags for the class.
        /// @param	doc			Documentation flags for the class.
        /// @param	host		Host flags for the class.

        explicit constexpr flags(behavior_flags behavior, documentation_flags doc, host_flags host = host_flags::none)
        : m_documentation{doc}
        , m_behavior{behavior}
        , m_host{host}
        {}


        /// Get the documentation flags of the class.
        /// @return	The documentation flags of the class.

        constexpr operator documentation_flags() const {
            return m_documentation;
        }


        /// Get the behavior flags of the class.
        /// @return	The behavior flags of the class.

        constexpr operator behavior_flags() const {
            return m_behavior;
        }


        /// Get the host flags of the class.
        /// @return	The host flags of the class.

        constexpr operator host_flags() const {
            return m_host;
        }

    private:
        const documentation_flags m_documentation{};
        const behavior_flags      m_behavior{};
        const host_flags          m_host{};
    };


    /// Declare flags for your class.
    /// These flags control documentation generation, host availability, and other behaviors.
    /// For argument options see the documentation on constructors for the #flags class.
    /// @see flags

    #define MIN_FLAGS const flags class_flags


    // SFINAE implementation used internally to determine if the Min class has
    // declared any class_flags using the classes and macros above.

    template<typename min_class_type>
    struct has_class_flags {
        template<class, class>
        class checker;

        template<typename C>
        static std::true_type test(checker<C, decltype(&C::class_flags)>*);

        template<typename C>
        static std::false_type test(...);

        typedef decltype(test<min_class_type>(nullptr)) type;
        static const bool value = is_same<std::true_type, decltype(test<min_class_type>(nullptr))>::value;
    };


    // Used internally.
    // Returns the declared class_flags if they exist.
    // The type T is one of the specialized flag types ( documentation_flags, host_flags, etc.).

    template<class min_class_type, class T>
    typename enable_if<has_class_flags<min_class_type>::value>::type class_get_flags(const min_class_type& instance, T& returned_flags) {
        returned_flags = instance.class_flags;
    }


    // Used internally.
    // Returns class_flags if they do *not* exist.
    // The type T is one of the specialized flag types ( documentation_flags, host_flags, etc.).

    template<class min_class_type, class T>
    typename enable_if<!has_class_flags<min_class_type>::value>::type class_get_flags(const min_class_type& instance, T& returned_flags) {
        returned_flags = T::none;
    }

}    // namespace c74::min
