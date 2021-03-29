/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include <sstream>
#include <unordered_map>

namespace c74::min {


    /// @defgroup attributes Attributes


    /// A callback function used by attributes that provide an optional customized "setter" routine.
    /// @ingroup attributes

    using setter = function;


    /// A callback function used by attributes that provide an optional customized "getter" routine.
    /// Typically this is provided to argument as a lamba function using the #MIN_GETTER_FUNCTION macro.
    /// @ingroup	attributes
    /// @return		A vector of atoms that represent the current state of the attribute.
    /// @see		MIN_GETTER_FUNCTION

    using getter = std::function<atoms()>;


    /// Provide the correct lamba function prototype for a getter parameter to the min::attribute constructor.
    /// @ingroup	attributes
    /// @see		getter
    /// @see		attribute

    #define MIN_GETTER_FUNCTION [this]() -> atoms


    /// A high-level meta-type that is associated with an attribute.
    /// These are used to enhance the experience of editing the attribute value using the inspector or attr ui.
    /// @ingroup attributes

    enum class style {
        none,           /// No special style.
        text,           /// Provide a text editor.
        onoff,          /// Edit using an on/off switch or toggle.
        enum_symbol,    /// Provide a list or menu of options, the actual stored attribute is a symbol.
        enum_index,     /// Provide a list or menu of options, the actual stored attribute is an int.
        rect,           /// Rectangular coordinate editor.
        font,           /// Provide a font dialog.
        file,           /// Provide a file chooser.
        time,           /// ITM time attributes many also specify a type of time (e.g. notes, bars-beats-units, etc.)
        color           /// Provide high-level color editors and swatches.
    };


    /// Symbolic names associated with the values of the enum min::style.
    /// @ingroup attributes
    /// @see style

    static std::unordered_map<style, symbol> style_symbols {
        {style::text, "text"},
        {style::onoff, "onoff"},
        {style::enum_symbol, "enum"},
        {style::enum_index, "enumindex"},
        {style::rect, "rect"},
        {style::font, "font"},
        {style::file, "file"},
        {style::color, "rgba"},
    };


    /// If the style of an attribute is one of the 'enum' types then an 'enum_map' maybe be
    /// supplied which provides symbolic names for each of the enum options.
    /// @ingroup attributes
    /// @see style

    using enum_map = std::vector<std::string>;


    /// Declare an attribute's visibility to the user.
    /// @ingroup attributes

    enum class visibility {
        show,      ///< standard behavior: show the attribute to the user
        hide,      ///< hide the attribute from the user
        disable    ///< don't create the attribute at all
    };


    /// Defines an attribute's category (group) in the inspector.
    /// @ingroup attributes

    using category = symbol;


    /// Defines attribute ordering in the inspector.
    /// A value zero means there is no special order and Max will take care of the ordering automatically.
    /// @ingroup attributes

    using order = int;


    /// The range provides a definition of acceptable or 'normal' values for an attribute.
    /// Unless specified as a template-parameter to the attribute, the range is only a suggestion to the user.
    /// Ranges for numeric types will be two numbers (a low bound and a high bound).
    /// For symbols or enums the range will provide the available options.
    /// @ingroup attributes

    using range = atoms;


    /// Defines whether an attribute is readonly (true) or if it is also writable (false).
    /// @ingroup attributes

    using readonly = bool;


    /// Defines a mapping to a color in Max for Live's themes.
    /// For examples of valid color names see the live.colors object in Max.
    /// @ingroup attributes

    class live_color {
		symbol m_color_name;
    public:
        live_color(const symbol& a_symbol)
        : m_color_name { a_symbol }
        {}

        operator symbol() const {
            return m_color_name;
        }
	};


    // Represents any type of attribute.
    // Used internally to allow heterogenous containers of attributes for the Min class.
    /// @ingroup attributes

    class attribute_base {
    protected:
        // Constructor. See the constructor documention for min::attribute<> to get more details on the arguments.

        attribute_base(object_base& an_owner, const std::string& a_name)
        : m_owner{an_owner}
        , m_name{a_name}
        , m_title{a_name}
        {}

    public:
        attribute_base(const attribute_base& other)  = delete;    // no copying allowed!
        attribute_base(const attribute_base&& other) = delete;    // no moving allowed!


        // All attributes must define what happens when you set their value.
        // Args may be modified if the range is constrained

        virtual attribute_base& operator=(atoms& args) = 0;


        // All attributes must define what happens when you set their value.
        // NOTE: args may be modified after this call due to range limiting behavior

        virtual void set(atoms& args, const bool notify = true, const bool override_readonly = false) = 0;


        // All attributes must define what happens when you get their value.

        virtual operator atoms() const = 0;


        // All attributes must define what happens when asked for their range of values.
        // The range must be in string format, values separated by spaces.

        virtual std::string range_string() const = 0;


        // All attributes must define what happens in the Max wrapper to
        // create the Max attribute and add it to the Max class.
        // Not intended for public use, but made a public member due to the
        // difficulty of making friends of the heavily templated SFINAE wrapper code.

        virtual void create(max::t_class* c, const max::method getter, const max::method setter, const bool isjitclass = 0) = 0;


        /// Determine the name of the datatype
        /// @return	The name of the datatype of the attribute.

        symbol datatype() const {
            return m_datatype;
        }


        /// Return the owner of the attribute
        /// @return	The owner of the attribute.

        object_base& owner() const {
            return m_owner;
        }


        /// Determine the name of the attribute
        /// @return	The name of the attribute.

        symbol name() const {
            return m_name;
        }


        /// Is the attribute writable (meaning settable)?
        /// @return	True if it is writable. Otherwise false.

        // note: cannot call it "readonly" because that is the name of a type and
        // it thus confuses the variadic template parsing below

        bool writable() const {
            return !m_readonly;
        }


        /// Is the attribute visible?
        /// @return The attribute's current visibility flag.

        visibility visible() const {
            return m_visibility;
        }


        /// Fetch the title/label as a string.
        /// This is how the name appears in the inspector.
        /// @return The attribute's label.
        ///			If the attribute has no label then the name of the object is used as the default.

        const char* label_string() const {
            return m_title;
        }


        /// Fetch the default value as a string.
        /// @return The attribute value as a string.

        virtual string default_string() const = 0;


        /// Return the provided description for use in documentation generation, auto-complete, etc.
        /// @return	The description string supplied when the attribute was created.

        std::string description_string() const {
            return m_description;
        }


        /// Return the style that is to be used for attribute editors such as the attrui object and the Max inspector.
        /// @return	The style supplied when the attribute was created.

        style editor_style() const {
            return m_style;
        }


        /// Return the category under which the attribute should appear in the Max inspector.
        /// @return	The category supplied when the attribute was created.

        symbol editor_category() const {
            return m_category;
        }


        /// Return the ordering priority for the attribute when listed in the inspector.
        /// @return	The order priority supplied when the attribute was created.

        int editor_order() const {
            return m_order;
        }


        /// Return the live color name if a mapping to a live color was defined for this attribute.
        /// @return The live color name or the empty symbol if no mapping was defined.

        symbol live_color_mapping() const { 
            return m_live_color;
        }

        /// Touch the attribute to force an update and notification of its value to any listeners.

        void touch() {
            max::object_attr_touch(m_owner, m_name);
        }

    protected:
        object_base& m_owner;
        symbol       m_name;
        symbol       m_title;
        symbol       m_datatype;
        setter       m_setter;
        getter       m_getter;
        bool         m_readonly { false };
        visibility   m_visibility { visibility::show };
        description  m_description;
		size_t       m_size {};                 // size of array/vector if attr is array/vector
        style        m_style { style::none };   // display style in Max
        symbol       m_category;                // Max inspector category
        int          m_order { 0 };             // Max inspector ordering
		symbol       m_live_color { k_sym__empty };

        // calculate the offset of the size member as required for array/vector attributes

        size_t size_offset() const {
            return (&m_size) - reinterpret_cast<size_t*>(&m_owner);
        }


        // return flags required by the max/obex attribute to get the correct behavior

        std::size_t flags(bool isjitclass) const {
            auto attrflags = 0;

            if (!writable()) {
                attrflags |= max::ATTR_SET_OPAQUE_USER;
                attrflags |= max::ATTR_SET_OPAQUE;
            }
            if (isjitclass) {
                attrflags |= max::ATTR_GET_DEFER_LOW;
                attrflags |= max::ATTR_SET_USURP_LOW;
            }
            return attrflags;
        }
    };


    // forward declarations of stuff implemented and documented further below...

    template<typename T, threadsafe threadsafety, template<typename> class limit_type, allow_repetitions repetitions>
    class attribute_threadsafe_helper;

    template<typename T, threadsafe threadsafety, template<typename> class limit_type, allow_repetitions repetitions>
    void attribute_threadsafe_helper_do_set(attribute_threadsafe_helper<T, threadsafety, limit_type, repetitions>* helper, atoms& args);


    /// An Attribute.
    /// Attributes in Max are how you create members whose state is addressable, queryable, and saveable.
    ///
    /// @ingroup	attributes
    /// @tparam		T				The type of the data saved in the attribute.
    ///								For example, `attribute<int>` will allocate and manage access to an int.
    /// @tparam		threadsafety	An optional parameter.
    ///								If your object has been written specifically and carefully to be threadsafe then
    ///								you may pass the option parameter threadsafe::yes.
    ///								The default is threadsafe::no, which is the correct choice in most cases.
    /// @tparam		limit_type		An optional parameter.
    ///								If your attribute is a numeric type (e.g. number or int), and it defines a range,
    ///								the class type you specify here will be used to limit the input values to that range.
    ///								The available options are the template classes defined in the #c74::min::limit namespace.
    ///								Namely: none, clamp, fold, and wrap.
    /// @see						buffer_index example object.

    template<typename T, threadsafe threadsafety, template<typename> class limit_type, allow_repetitions repetitions>
    class attribute : public attribute_base {
    private:
        // constructor utility: handle an argument defining an attribute's title / label

        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, title>::value>::type assign_from_argument(const argument_type& arg) noexcept {
            m_title = arg;
        }


        // constructor utility: handle an argument defining an attribute's description

        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, description>::value>::type assign_from_argument(const argument_type& arg) noexcept {
            m_description = arg;
        }


        // constructor utility: handle an argument defining a attribute's range

        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, range>::value>::type assign_from_argument(const argument_type& arg) noexcept {
            m_range_args = arg;
        }


        // constructor utility: handle an argument defining an enum mapping to associate strings with enum constants
        // this is used in place of the range for index enum attributes.

        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, enum_map>::value>::type assign_from_argument(const argument_type& arg) noexcept {
            m_enum_map = arg;
        }


        // constructor utility: handle an argument defining an attribute's setter function

        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, setter>::value>::type assign_from_argument(const argument_type& arg) noexcept {
            const_cast<argument_type&>(m_setter) = arg;
        }

        // constructor utility: handle an argument defining an attribute's getter function

        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, getter>::value>::type assign_from_argument(const argument_type& arg) noexcept {
            m_getter = arg;
        }


        // constructor utility: handle an argument defining an attribute's readonly property

        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, readonly>::value>::type assign_from_argument(const argument_type& arg) noexcept {
            m_readonly = arg;
        }


        // constructor utility: handle an argument defining an attribute's visibility property

        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, visibility>::value>::type assign_from_argument(const argument_type& arg) noexcept {
            m_visibility = arg;
        }


        // constructor utility: handle an argument defining an attribute's style property

        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, style>::value>::type assign_from_argument(const argument_type& arg) noexcept {
            m_style = arg;
        }


        // constructor utility: handle an argument defining an attribute's category property

        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, category>::value>::type assign_from_argument(const argument_type& arg) noexcept {
            m_category = arg;
        }


        // constructor utility: handle an argument defining an attribute's order property

        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, order>::value>::type assign_from_argument(const argument_type& arg) noexcept {
           m_order = arg;
        }

        
        // constructor utility: handle an argument defining an attribute's live_color property

		template<typename argument_type>
		constexpr typename enable_if<is_same<argument_type, live_color>::value>::type assign_from_argument(const argument_type& arg) noexcept {
			m_live_color = static_cast<symbol>(arg);
		}


        // constructor utility: empty argument handling (required for handling recursive variadic templates)

        constexpr void handle_arguments() noexcept {
            ;
        }


        // constructor utility: handle N arguments of any type by recursively working through them
        //	and matching them to the type-matched routine above.

        template<typename FIRST_ARG, typename... REMAINING_ARGS>
        constexpr void handle_arguments(FIRST_ARG const& first, REMAINING_ARGS const&... args) noexcept {
            assign_from_argument(first);
            if (sizeof...(args) > 0)
                handle_arguments(args...);    // recurse
        }

    public:
        /// Create an attribute.
        /// @param an_owner			The instance pointer for the owning C++ class, typically you will pass 'this'
        /// @param a_name			A string specifying the name of the attribute when dynamically addressed or inspected.
        /// @param a_default_value	The default value of the attribute, which will be set when the instance is created.
        /// @param args				N arguments specifying optional properties of an attribute such as setter, label, style, etc.

        template<typename... ARGS>
        attribute(object_base* an_owner, const std::string a_name, const T a_default_value, ARGS... args);


        attribute(const attribute& other)  = delete;    // no copying allowed!
        attribute(const attribute&& other) = delete;    // no moving allowed!


        // DO NOT USE
        // This is an internal method used to
        // create the peer Max attribute and add it to the Max class.
        // It is made 'public' due to the trickiness of the SFINAE-enabled templated functions which call this from the wrapper.

        void create(max::t_class* c, const max::method getter, const max::method setter, bool isjitclass = 0) override;


        /// Get the range of the attribute.
        /// @return	For numeric types return the low and high bounds.
        ///			For enums return the names of each of the options.

        std::vector<T>& range_ref() {
            return m_range;
        }


        // DO NOT USE
        // This is an internal method used to fetch the range in string format when creating the peer Max attribute.
        // It is made 'public' due to the trickiness of the SFINAE-enabled templated functions which call this from the wrapper.

        std::string range_string() const override;


        // DO NOT USE
        // This is an internal method used to fetch the range as provided to the attribute declaration in the Min object.
        // It will be copied out using a SFINAE-enabled templated helper.
        // You can the range after it is copied using the range_ref() method.

        atoms get_range_args() {
            return m_range_args;
        }


        string default_string() const override {
            auto as = to_atoms(m_default);
            auto s = to_string(as);
            return s;
        }


        // Returns the strings associated with an enum attribute.
        // Only available for enum attributes.
        // @return A copy of the enum_map.
        //
        // This is used by the range copying at setup.
        // It isn't clear that it is actually useful outside of this context, so not officially documenting it.

        template<class U = T, typename enable_if<is_enum<U>::value, int>::type = 0>
        enum_map get_enum_map() const {
            return m_enum_map;
        }


        /// Set the attribute value using the native type of the attribute.
        /// @param	arg		The new value to be assigned to the attribute.

        attribute& operator=(const T arg) {
            atoms as = {atom(arg)};
            *this    = as;
            return *this;
        }


        /// Set the attribute value using atoms.
        /// @param	args	The new value to be assigned to the attribute.

        attribute& operator=(atoms& args) override {
           set(args);
            return *this;
        }


        // DO NOT USE
		// This is an internal method
        //
        // This exists because MIN_FUNCTION, which is used by every attribute setter in user code,
        // declares its args to be const to make it clear to users that they shouldn't be changing the args.
        // We are priviledged in this scenario because we made the atoms as a copy of what Max provided us and
        // we know it is safe to use them -- and potentially modify them when range limiting is applied.
        //
        // For user code, please use the version above and pass mutable atoms
        
        attribute& operator=(const atoms& args) {
            set(const_cast<atoms&>(args));
            return *this;
        }


        // special setter for enum attributes
        // converts from the name to the index and then calls the above assignment operator

        template<class U = T, typename enable_if<is_enum<U>::value, int>::type = 0>
        attribute& operator=(const symbol arg) {
            for (auto i = 0; i < m_enum_map.size(); ++i) {
                if (arg == m_enum_map[i]) {
                    *this = static_cast<T>(i);
                    break;
                }
            }
            return *this;
        }


        /// Set the attribute value using atoms.
        /// Permits additional control as compared with using the assignment operators.
        /// @param	args				The new value to be assigned to the attribute.
        ///                             May be modified by this call when optional range-limiting is applied.
        /// @param	notify				Notify the Max environment when the attribute is set.
        ///								This is performed by setting the attribute value using the standard Max API call.
        ///								If you are setting the value internally to your class you may wish to turn this off to reduce computational
        ///costs.
        /// @param	override_readonly	Normally a readonly attribute cannot be written (assigned a value).
        ///								Setting this to true will allow you to override the readonly flag and set the attribute value
        ///anyway.

        void set(atoms& args, const bool notify = true, const bool override_readonly = false) override {
            if (!writable() && !override_readonly)
                return;    // we're all done... unless this is a readonly attr that we are forcing to update

            if (repetitions == allow_repetitions::no && compare_to_current_value(args))
                return;

#ifndef MIN_TEST    // At this time the Mock Kernel does not implement object_attr_setvalueof(), so we can't use it for unit tests
            if (notify && this_class) {    // Use the Max API to set the attribute value
                max::object_attr_setvalueof(
                    m_owner, m_name, static_cast<long>(args.size()), static_cast<const c74::max::t_atom*>(&args[0]));
            }
            else
#endif           // !MIN_TEST
            {    // Set the value ourselves
                // currently all jitter attributes bypass the defer mechanism here opting to instead use the default jitter handling
                // were we to simply call `m_helper.set(args);` then our defer mechanism would be called **in addition to** jitter's
                // deferring

                if (m_owner.is_jitter_class())
                    attribute_threadsafe_helper_do_set<T, threadsafety>(&m_helper, args);
                else
                    m_helper.set(args);
            }

            if (name() == k_sym_value)
                c74::max::object_notify(owner().maxobj(), k_sym_modified, nullptr);
        }


        /// Get the raw attribute value from an attribute.
        /// @return The attribute value.

        T& get() {
            return m_value;
        }


        /// Compare a value against the attribute's current value.
        /// @param	lhs		The attribute
        /// @param	rhs		The value to compare against the attribute
        /// @return			True if they are the same. Otherwise false.

        friend bool operator==(const attribute& lhs, const T& rhs) {
            return lhs.m_value == rhs;
        }


        /// Get the attribute value as a vector of atoms.
        /// @return	The value of the attribute.

        operator atoms() const override {
            if (m_getter)
                return m_getter();
            else
                return to_atoms(m_value);
        }


        /// Get the attribute value as a const reference to the native datatype.
        /// We need to return by const reference in cases where the type of the attribute is a class.
        /// For example, a #time_value attribute cannot be copy constructed.
        /// @return	The value of the attribute.

        operator const T&() const {
            if (m_getter)
                assert(false);    // at the moment there is no easy way to support this
            return m_value;
        }


        /// Get the attribute value as a reference to the native datatype.
        /// Getting a writable reference to the underlying data is of particular importance
        /// for e.g. vector<number> attributes.
        /// @return	The value of the attribute.

        operator T&() {
            if (m_getter)
                assert(false);    // at the moment there is no easy way to support this
            return m_value;
        }


        // simplify getting millisecond time from a time_value attribute

        template<class U = T, typename enable_if<is_same<U, time_value>::value, int>::type = 0>
        operator double() const {
            return m_value;
        }


        /// Get a component of an attribute value when that attribute is a vector of numbers.
        /// @param	index	The index of the item in the vector to access.
        /// @return			A writable reference to the value at an index of the attribute.

        template<class U = T, typename enable_if<is_same<U, numbers>::value, int>::type = 0>
        double& operator[](const size_t index) {
            return m_value[index];
        }

        /// Get a component of an attribute value when that attribute is a vector of ints.
        /// @param    index     The index of the item in the vector to access.
        /// @return          A writable reference to the value at an index of the attribute.

        template<class U = T, typename enable_if<is_same<U, ints>::value, int>::type = 0>
        int& operator[](const size_t index) {
            return m_value[index];
        }


        /// Is the attribute currently disabled?
        /// @return	True if it is disabled. False if it is active.

        bool disabled() const {
            return c74::max::object_attr_getdisabled(m_owner, m_name);
        }


        /// Disable the attribute.
        /// This will result in the attribute being "grayed-out" in the inspector.
        /// @param	value	Pass true to disable the attribute. Otherwise pass false to make it active.

        void disable(const bool value) {
            c74::max::object_attr_setdisabled(m_owner, m_name, value);
        }


    private:
        T              m_value;         // The actual data wrapped by this attribute.
        T              m_default;       // The default value for this attribute.
        atoms          m_range_args;    // The range/enum as provided by the owning Min object.
        std::vector<T> m_range;         // The range/enum translated into the native datatype.
        enum_map       m_enum_map;      // The enum mapping for indexed enums (as opposed to symbol enums).
        attribute_threadsafe_helper<T, threadsafety, limit_type, repetitions> m_helper{this};    // Attribute setting implementation for the specified threadsafety.

        friend void attribute_threadsafe_helper_do_set<T, threadsafety, limit_type, repetitions>(attribute_threadsafe_helper<T, threadsafety, limit_type, repetitions>* helper, atoms& args);


        // Copy m_range_args to m_range when the attribute is created.
        // Implemented in c74_min_attribute_impl.h.

        void copy_range();


        // Implemented in c74_min_attribute_impl.h.

        bool compare_to_current_value(const atoms& args) const;


        // Apply range limiting to all numerical types.
        // Optimization for the most common case: no limiting at all.

        template<class U = T, typename enable_if<is_same<limit_type<U>, limit::none<U>>::value, int>::type = 0>
        void constrain(atoms& args) {
            // no limiting, so do nothing
        }


        // Apply range limiting to all numerical types (except enums).
        // Note that enums are already range-limited within the min::atom.

        template<class U = T, typename enable_if<!is_same<limit_type<U>, limit::none<U>>::value, int>::type = 0>
        void constrain(atoms& args) {
            // TODO: type checking on the above so that it is not applied to vectors or colors
            args[0] = limit_type<T>::apply(args[0], m_range[0], m_range[1]);
        }


        // Assign the value to the internal data storage member.
        // Occurs after the limits are constrained, the setter is called, etc.

        template<class U = T, typename enable_if<!is_enum<U>::value, int>::type = 0>
        void assign(const atoms& args) {
            m_value = from_atoms<T>(args);
        }


        // Assign the value to the internal data storage member
        // when the attribute type is an enum.
        // Allows users to specify the symbolic name and maps it to the underlying int.

        template<class U = T, typename enable_if<is_enum<U>::value, int>::type = 0>
        void assign(const atoms& args) {
            const atom& a = args[0];

            if (a.a_type == max::A_SYM) {
                for (auto i = 0; i < m_enum_map.size(); ++i) {
                    if (a == m_enum_map[i]) {
                        m_value = static_cast<T>(i);
                        break;
                    }
                }
            }
            else
                m_value = from_atoms<T>(args);
        }
    };


#ifdef MAC_VERSION
#pragma mark -
#pragma mark Threadsafe Helper
#endif

    // Regarding thread-safety...
    //
    // First, lets just consider the getter.
    // We can't defer here -- it must happen at a higher level (in Max somewhere).
    // Perhaps we could lock Max with a critical region but that is fraught with risks and hazards.
    // This is because we are required to return the value syncronously.
    // So getters are out of our control.
    //
    // Second, let's consider the setter.
    // We can defer this (and should if the setter is not threadsafe).
    //
    // If we defer, we need to do it in the attribut<>::set() method
    // because it is common to set the attr value from calls other than just the outside Max call.
    // Unfortunately, we cannot do a partial template specialization for a member function in C++.
    // So the set method is then required to call a templated class (which can be partially specialized) as a functor.
    //
    // That is what we have in attribute_threadsafe_helper.


    // Shared code used by all of the various incarnations of attribute_threadsafe_helper
    // to set an attribute.

    // args may be modified as a side-effect of calling this method (e.g. for range limiting)

    template<typename T, threadsafe threadsafety, template<typename> class limit_type, allow_repetitions repetitions>
    void attribute_threadsafe_helper_do_set(attribute_threadsafe_helper<T, threadsafety, limit_type, repetitions>* helper, atoms& args) {
        auto& attr = *helper->m_attribute;

        attr.constrain(args);

        if (attr.m_setter)
            attr.m_value = from_atoms<T>(attr.m_setter(args, -1));
        else
            attr.assign(args);
    }


    // A version of attribute_threadsafe_helper<> for attributes which are threadsafe.
    // This is the simplest case:
    // the author of the owning object told us it is threadsafe and so we trust them that we
    // don't need to do anything special.

    template<typename T, template<typename> class limit_type, allow_repetitions repetitions>
    class attribute_threadsafe_helper<T, threadsafe::yes, limit_type, repetitions> {
        friend void attribute_threadsafe_helper_do_set<T, threadsafe::yes, limit_type>(attribute_threadsafe_helper<T, threadsafe::yes, limit_type, repetitions>* helper, atoms& args);

    public:
        explicit attribute_threadsafe_helper(attribute<T, threadsafe::yes, limit_type, repetitions>* an_attribute)
        : m_attribute(an_attribute)
        {}

        void set(atoms& args) {
            attribute_threadsafe_helper_do_set(this, args);
        }

    private:
        attribute<T, threadsafe::yes, limit_type, repetitions>* m_attribute;
    };


    // C-callback for the qelem used to defer attribute setting to the main thread
    // for thread-unsafe attributes.

    template<typename T, threadsafe threadsafety, template<typename> class limit_type, allow_repetitions repetitions>
    void attribute_threadsafe_helper_qfn(attribute_threadsafe_helper<T, threadsafety, limit_type, repetitions>* helper) {
        static_assert(threadsafety != threadsafe::yes, "helper function should not be called by threadsafe attrs");
        attribute_threadsafe_helper_do_set<T, threadsafety, limit_type, repetitions>(helper, helper->m_value);
    }


    // A version of attribute_threadsafe_helper<> for attributes which are not known to be threadsafe.
    // These will check all setter calls to ensure that they are on the main thread.
    // If they are not then defer the setter calls to the main thread using a qelem.

    template<typename T, template<typename> class limit_type, allow_repetitions repetitions>
    class attribute_threadsafe_helper<T, threadsafe::no, limit_type, repetitions> {
        friend void attribute_threadsafe_helper_do_set<T, threadsafe::no, limit_type>(attribute_threadsafe_helper<T, threadsafe::no, limit_type, repetitions>* helper, atoms& args);
        friend void attribute_threadsafe_helper_qfn<T, threadsafe::no, limit_type, repetitions>(attribute_threadsafe_helper<T, threadsafe::no, limit_type, repetitions>* helper);

    public:
        explicit attribute_threadsafe_helper(attribute<T, threadsafe::no, limit_type, repetitions>* an_attribute)
        : m_attribute(an_attribute) {
            m_qelem = (max::t_qelem*)max::qelem_new(this, (max::method)attribute_threadsafe_helper_qfn<T, threadsafe::no, limit_type, repetitions>);
        }

        ~attribute_threadsafe_helper() {
            max::qelem_free(m_qelem);
        }

        void set(atoms& args) {
            if (max::systhread_ismainthread())
                attribute_threadsafe_helper_do_set(this, args);
            else {
                m_value = args;
                max::qelem_set(m_qelem);
            }
        }

    private:
        attribute<T, threadsafe::no, limit_type, repetitions>*  m_attribute;
        max::t_qelem*                                           m_qelem;
        atoms                                                   m_value;
    };


    // A version of attribute_threadsafe_helper<> for attributes which inherit the threadsafety declaration from thier owning class.
    // These will check all setter calls to ensure that they are on the main thread -- or that they are declared as threadsafe.
    // If they are not then defer the setter calls to the main thread using a qelem.

    template<typename T, template<typename> class limit_type, allow_repetitions repetitions>
    class attribute_threadsafe_helper<T, threadsafe::undefined, limit_type, repetitions> {
        friend void attribute_threadsafe_helper_do_set<T, threadsafe::undefined, limit_type, repetitions>(attribute_threadsafe_helper<T, threadsafe::undefined, limit_type, repetitions>* helper, atoms& args);
        friend void attribute_threadsafe_helper_qfn<T, threadsafe::undefined, limit_type, repetitions>(attribute_threadsafe_helper<T, threadsafe::undefined, limit_type, repetitions>* helper);

    public:
        explicit attribute_threadsafe_helper(attribute<T, threadsafe::undefined, limit_type, repetitions>* an_attribute)
        : m_attribute(an_attribute) {
            m_qelem = (max::t_qelem*)max::qelem_new(this, (max::method)attribute_threadsafe_helper_qfn<T, threadsafe::no, limit_type, repetitions>);
        }

        ~attribute_threadsafe_helper() {
            max::qelem_free(m_qelem);
        }

        void set(atoms& args) {
            if (m_attribute->owner().is_assumed_threadsafe() || max::systhread_ismainthread())
                attribute_threadsafe_helper_do_set(this, args);
            else {
                m_value = args;
                max::qelem_set(m_qelem);
            }
        }

    private:
        attribute<T, threadsafe::undefined, limit_type, repetitions>*   m_attribute;
        max::t_qelem*                                                   m_qelem;
        atoms                                                           m_value;
    };


#ifdef MAC_VERSION
#pragma mark -
#pragma mark Wrapper methods
#endif


    //	Native Max methods for the wrapper class to perform getting of attribute values

    template<class T>
    max::t_max_err min_attr_getter(minwrap<T>* self, max::t_object* maxattr, long* ac, max::t_atom** av) {
        const symbol	attr_name	= static_cast<const max::t_symbol*>(max::object_method(maxattr, k_sym_getname));
        auto&	        attr		= self->m_min_object.attributes()[attr_name.c_str()];
        atoms	        rvals		= *attr;

        if ((*ac) != rvals.size() || !(*av)) {		 // otherwise use memory passed in
            if (*ac && *av) {
                sysmem_freeptr(*av);
                *av = NULL;
            }
            *ac = static_cast<long>(rvals.size());
            *av = reinterpret_cast<max::t_atom*>( max::sysmem_newptr(sizeof(max::t_atom) * (*ac)) );
        }
		assert(*av);

        for (auto i=0; i<(*ac); ++i)
            (*av)[i] = rvals[i];

        return 0;
    }


    //	Native Max methods for the wrapper class to perform setting of attribute values

    template<class T>
    max::t_max_err min_attr_setter(minwrap<T>* self, max::t_object* maxattr, const long ac, const max::t_atom* av) {
		const symbol attr_name { static_cast<const max::t_symbol*>(max::object_method(maxattr, k_sym_getname)) };
		auto         attr      { self->m_min_object.attributes()[attr_name.c_str()] };

        if (attr) {
			const atom_reference args(ac, const_cast<max::t_atom*>(av)); // atom_reference cannot guarantee constness, but we are only using it copy atoms out on the line below
            atoms as(args.begin(), args.end());
            attr->set(as, false, false);
        }
        return 0;
    }

}    // namespace c74::min
