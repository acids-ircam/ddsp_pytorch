/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {

    
    #define MIN_TAGS static constexpr const char* class_tags

    using tags = std::vector<std::string>;

    template<typename min_class_type>
    struct has_class_tags {
        template<class, class>
        class checker;

        template<typename C>
        static std::true_type test(checker<C, decltype(&C::class_tags)>*);

        template<typename C>
        static std::false_type test(...);

        typedef decltype(test<min_class_type>(nullptr)) type;
        static const bool value = is_same<std::true_type, decltype(test<min_class_type>(nullptr))>::value;
    };

    template<class min_class_type>
    typename enable_if<has_class_tags<min_class_type>::value>::type get_tags(tags& returned_tags) {
        returned_tags = str::split(min_class_type::class_tags, ',');
    }

    template<class min_class_type>
    typename enable_if<!has_class_tags<min_class_type>::value>::type get_tags(tags& returned_tags) {
        returned_tags = {};
    }


    // SFINAE implementation used internally to determine if the Min class has a member named mousedragdelta.
    // NOTE: This relies on the C++ member name being "mousedragdelta" -- not just the Max message name being "mousedragdelta".
    //
    // To test this in isolation, for a class named slide, use the following code:
    // static_assert(has_mousedragdelta<slide>::value, "error");

    template<typename min_class_type>
    struct has_mousedragdelta {
        template<class, class>
        class checker;

        template<typename C>
        static std::true_type test(checker<C, decltype(&C::mousedragdelta)>*);

        template<typename C>
        static std::false_type test(...);

        typedef decltype(test<min_class_type>(nullptr)) type;
        static const bool value = is_same<std::true_type, decltype(test<min_class_type>(nullptr))>::value;
    };

    // An alternative to the above to all "m_mousedragdelta" in addition to "mousedragdelta"

    template<typename min_class_type>
    struct has_m_mousedragdelta {
        template<class, class>
        class checker;

        template<typename C>
        static std::true_type test(checker<C, decltype(&C::m_mousedragdelta)>*);

        template<typename C>
        static std::false_type test(...);

        typedef decltype(test<min_class_type>(nullptr)) type;
        static const bool value = is_same<std::true_type, decltype(test<min_class_type>(nullptr))>::value;
    };


    // SFINAE implementation used internally to determine if the Min class has a member named focusgained.
    // NOTE: This relies on the C++ member name being "focusgained" -- not just the Max message name being "focusgained".
    //
    // To test this in isolation, for a class named slide, use the following code:
    // static_assert(has_focusgained<slide>::value, "error");

    template<typename min_class_type>
    struct has_focusgained {
        template<class, class>
        class checker;

        template<typename C>
        static std::true_type test(checker<C, decltype(&C::focusgained)>*);

        template<typename C>
        static std::false_type test(...);

        typedef decltype(test<min_class_type>(nullptr)) type;
        static const bool value = is_same<std::true_type, decltype(test<min_class_type>(nullptr))>::value;
    };

    // An alternative to the above to all "m_focusgained" in addition to "focusgained"

    template<typename min_class_type>
    struct has_m_focusgained {
        template<class, class>
        class checker;

        template<typename C>
        static std::true_type test(checker<C, decltype(&C::m_focusgained)>*);

        template<typename C>
        static std::false_type test(...);

        typedef decltype(test<min_class_type>(nullptr)) type;
        static const bool value = is_same<std::true_type, decltype(test<min_class_type>(nullptr))>::value;
    };


    
    /// The base class for all first-class objects that are to be exposed in the Max environment.
    ///
    /// We pass the class type as a template parameter to this class.
    /// This allows for [static polymorphism](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern#Static_polymorphism).
    /// One benefits of this are leveraged when instantiating class instances directly instead of through the Max interface,
    /// such as when unit testing or embedding an object inside of another object.
    ///
    /// @tparam min_class_type	The name of your class that is extenting min::object.
    /// @tparam threadsafety	The default threadsafety assumption for all messages and attributes in this class.

    template<class min_class_type, threadsafe threadsafety = threadsafe::no>
    class object : public object_base {
    public:
        /// Constructor.

        object() {

            // The way objects are created for the Max environment requires that memory be allocated first
            // using object_alloc() or jit_object_alloc(), which is followed by the use of placement-new to contruct the C++ class.
            //
            // When this occurs the m_maxobj member is already set prior to the constructor being run.
            // If there is no valid m_maxobj then that means this class was created outside of the Max environment.
            //
            // This could occur if a class uses another class directly or in the case of unit testing.
            // In such cases we need to do something reasonable so that our invariants can be held true
        }

        /// Destructor.

        virtual ~object()
        {}


        bool is_jitter_class() const override {
            return is_base_of<matrix_operator_base, min_class_type>::value;
        };

        bool is_ui_class() const override {
            return is_base_of<ui_operator_base, min_class_type>::value;
        }

        bool has_mousedragdelta() const override {
            return c74::min::has_mousedragdelta<min_class_type>::value || has_m_mousedragdelta<min_class_type>::value;
        }

        bool is_focusable() const override {
            return c74::min::has_focusgained<min_class_type>::value || has_m_focusgained<min_class_type>::value;
        }

        bool is_assumed_threadsafe() const override {
            return threadsafety == threadsafe::yes;
        }

        virtual strings tags() const override {
            strings t;
            get_tags<min_class_type>(t);
            for (auto& a_tag : t)
                a_tag = str::trim(a_tag);
            return t;
        }



    protected:
        logger cout     { this, logger::type::message };
        logger cwarn    { this, logger::type::warning };
        logger cerr     { this, logger::type::error };
    };

}    // namespace c74::min
