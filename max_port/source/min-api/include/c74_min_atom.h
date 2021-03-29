/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {

    class event;

    class atom : public max::t_atom {
    public:
        /// Default constructor -- an empty atom
        atom() {
            this->a_type    = c74::max::A_NOTHING;
            this->a_w.w_obj = nullptr;
        }

        /// constructor with generic initializer
        template<class T, typename enable_if<!std::is_enum<T>::value && !is_same<T, std::vector<atom>>::value, int>::type = 0>
        atom(const T initial_value) {
            *this = initial_value;
        }

        /// constructor with enum initializer
        template<class T, typename enable_if<std::is_enum<T>::value, int>::type = 0>
        atom(const T initial_value) {
            *this = static_cast<max::t_atom_long>(initial_value);
        }

        atom(const event& e) {
            this->a_type    = c74::max::A_OBJ;
            this->a_w.w_obj = const_cast<max::t_object*>(reinterpret_cast<const max::t_object*>(&e));
        }


        // copy/move constructors and assignment
        atom(const atom& source) = default;
        atom(atom&& source)      = default;
        atom& operator=(const atom& source) = default;
        atom& operator=(atom&& source) = default;


        atom& operator=(const max::t_atom& value) {
            this->a_type = value.a_type;
            this->a_w    = value.a_w;
            return *this;
        }

        atom& operator=(const max::t_atom* init) {
            *this = *init;
            return *this;
        }

        atom& operator=(const int64_t value) {
            atom_setlong(this, value);
            return *this;
        }

        atom& operator=(const int32_t value) {
            atom_setlong(this, value);
            return *this;
        }

        atom& operator=(const bool value) {
            atom_setlong(this, value);
            return *this;
        }

        atom& operator=(const double value) {
            atom_setfloat(this, value);
            return *this;
        }

        atom& operator=(const max::t_symbol* value) {
            atom_setsym(this, value);
            return *this;
        }

        atom& operator=(const symbol value) {
            atom_setsym(this, value);
            return *this;
        }

        atom& operator=(const char* value) {
            atom_setsym(this, symbol(value));
            return *this;
        }

        atom& operator=(const std::string& value) {
            max::atom_setsym(this, symbol(value.c_str()));
            return *this;
        }

        atom& operator=(max::t_object* value) {
            atom_setobj(this, static_cast<void*>(value));
            return *this;
        }

        atom& operator=(max::t_class* value) {
            atom_setobj(this, static_cast<void*>(value));
            return *this;
        }

        atom& operator=(const void* value) {
            atom_setobj(this, const_cast<void*>(value));
            return *this;
        }

        atom& operator=(void* value) {
            atom_setobj(this, value);
            return *this;
        }


        /// Enum assigning constructor
        template<class T, typename enable_if<std::is_enum<T>::value, int>::type = 0>
        operator T() const {
            return static_cast<T>(atom_getlong(this));
        }

        operator float() const {
            return static_cast<float>(atom_getfloat(this));
        }

        operator double() const {
            return atom_getfloat(this);
        }

        operator int() const {
            return static_cast<int>(atom_getlong(this));
        }

        operator long() const {
            return static_cast<long>(atom_getlong(this));
        }

        operator long long() const {
            return static_cast<long long>(atom_getlong(this));
        }

        operator size_t() const {
            return static_cast<size_t>(atom_getlong(this));
        }

        operator bool() const {
            return atom_getlong(this) != 0;
        }

        operator max::t_symbol*() const {
            return atom_getsym(this);
        }

        operator max::t_object*() const {
            return static_cast<max::t_object*>(atom_getobj(this));
        }

        operator max::t_class*() const {
            return static_cast<max::t_class*>(atom_getobj(this));
        }

        operator void*() const {
            return atom_getobj(this);
        }

        operator std::string() const {
            std::string s;

            switch (a_type) {
                case max::A_SEMI:
                    s = ";";
                    break;
                case max::A_COMMA:
                    s = ",";
                    break;
                case max::A_SYM:
                    s = a_w.w_sym->s_name;
                    break;
                case max::A_OBJ:
                    if (a_w.w_obj)
                        s = c74::max::object_classname(a_w.w_obj)->s_name;
                    else
                        s = "NULL";
                    break;
                case max::A_LONG:
                    s = std::to_string(a_w.w_long);
                    break;
                case max::A_FLOAT:
                    s = std::to_string(a_w.w_float);
                    break;
                default:
                    s = "?";
                    break;
            }

            return s;
        }


        /// Compare an atom against a value for equality.
        bool operator==(const max::t_symbol* s) const;

        /// Compare an atom against a value for equality.
        bool operator==(const symbol s) const;

        /// Compare an atom against a value for equality.
        bool operator==(const char* str) const;

        /// Compare an atom against a value for equality.
        bool operator==(const bool value) const;

        /// Compare an atom against a value for equality.
        bool operator==(const int value) const;

        /// Compare an atom against a value for equality.
        bool operator==(const long value) const;

        /// Compare an atom against a value for equality.
        bool operator==(const double value) const;

        /// Compare an atom against a value for equality.
        bool operator==(const max::t_object* value) const;

        /// Compare an atom against an atom for equality.
        bool operator==(const max::t_atom& b) const;

        /// Compare an atom against an atom for equality.
        bool operator==(const time_value value) const;


        /// Return the type of the data contained in the atom.
        message_type type() const {
            return static_cast<message_type>(a_type);
        }

    };


    /// The atoms container is the standard means by which zero or more values are passed.
    /// It is implemented as a std::vector of the atom type, and thus atoms contained in an
    /// atoms container are 'owned' copies... not simply a reference to some externally owned atoms.

    // TODO: how to document inherited interface, e.g. size(), begin(), etc. ?

    using atoms = std::vector<atom>;


#ifdef __APPLE__
#pragma mark -
#pragma mark AtomRef
#endif


    /// The atom_reference type defines a container for atoms by reference, providing an interface
    /// that is interoperable with any of the classic standard library containers.
    ///
    /// Typically you *do not use* the atom_reference type explicitly.
    /// It is rather intended as an intermediary between the atoms container type and
    /// old C-style functions in the Max API.
    /// As such it resembles some of the aims of the gsl::span type but serving a much more specialized purpose.
    ///
    /// IMPORTANT: The size and order of members in this class are designed to make it a drop-in replace for
    /// the old C-style argc/argv pairs!  As such, no changes or additions should be made with regards to
    /// members, virtual methods, etc.

    class atom_reference {
    public:
        using size_type      = long;
        using value_type     = max::t_atom*;
        using iterator       = max::t_atom*;
        using const_iterator = const max::t_atom*;

        iterator begin() {
            return m_av;
        }
        const_iterator begin() const {
            return m_av;
        }
        iterator end() {
            return m_av + m_ac;
        }
        const_iterator end() const {
            return m_av + m_ac;
        }

        size_type size() const {
            return m_ac;
        }
        bool empty() const {
            return size() == 0;
        }

        // We don't own the array of atoms, so we cannot do these operations:
        // insert
        // erase
        // push_back
        // push_front
        // pop_front
        // pop_back

        // TODO: we could consider implementing the following (but we may not need them due to limited role of this type):
        // front()
        // back()
        // operator []
        // at()

        // The ctor does not alter the atoms, 
        // but note that some future operations done to the atom_reference could unless it is a const atom_reference

        atom_reference(const long argc, const max::t_atom* argv)
        : m_ac { argc }
        , m_av { const_cast<max::t_atom*>(argv) }
        {}

        
        atom_reference& operator=(const symbol& value) {
            m_ac = 1;
            atom_setsym(m_av, value);
            return *this;
        }

        atom_reference& operator=(const int value) {
            m_ac = 1;
            atom_setlong(m_av, value);
            return *this;
        }

        atom_reference& operator=(const long value) {
            m_ac = 1;
            atom_setlong(m_av, value);
            return *this;
        }

        atom_reference& operator=(const double value) {
            m_ac = 1;
            atom_setfloat(m_av, value);
            return *this;
        }

        atom_reference& operator=(const max::t_object* value) {
            m_ac = 1;
            atom_setobj(m_av, const_cast<max::t_object*>(value));
            return *this;
        }


        operator atom() const {
            if (empty())
                throw std::out_of_range("atomref is empty");
            return atom(m_av);
        }

        operator atoms() const {
            atoms as(m_ac);

            for (auto i = 0; i < m_ac; ++i)
                as[i] = m_av + i;
            return as;
        }


        operator vector<int>() const {
            vector<int> v(m_ac);
            for (auto i = 0; i < m_ac; ++i)
                v[i] = static_cast<int>( atom_getlong(m_av + i) );
            return v;
        }


    private:
        long         m_ac;
        max::t_atom* m_av;
    };

}    // namespace c74::min


#ifdef __APPLE__
#pragma mark -
#pragma mark Utilities
#endif


namespace std {

    /// overload of the std::to_string() function for the min::atoms type
    // it is perfectly legal to make this overload in the std namespace because it is overloaded on our user-defined type
    // as stated in section 17.6.4.2.1 of working draft version N4296 of the C++ Standard at
    // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4296.pdf

    inline string to_string(const c74::min::atoms& as) {
        long   textsize = 0;
        char*  text     = nullptr;
        string str;

        auto err = c74::max::atom_gettext((long)as.size(), &as[0], &textsize, &text, c74::max::OBEX_UTIL_ATOM_GETTEXT_SYM_NO_QUOTE);
        if (!err)
            str = text;
        else
            c74::max::object_error(nullptr, "problem geting text from atoms");

        if (text && textsize)
            c74::max::sysmem_freeptr(text);

        return str;
    }


    /// overload of the std::to_string() function for the min::atom_reference type

    inline string to_string(const c74::min::atom_reference& ar) {
        c74::min::atoms as;
        for (const auto& ref : ar)
            as.push_back(ref);
        return to_string(as);
    }

}    // namespace std


namespace c74::min {

    /// Expose atom for use in std output streams.
    template<class charT, class traits>
    std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& stream, const c74::min::atom& a) {
        return stream << std::string(a);
    }


    /// Copy values from any STL container to a vector of atoms
    /// @tparam	T			The type of the container
    /// @param	container	The container instance whose values will be copied
    /// @return				A vector of atoms

    template<class T, typename enable_if<!is_symbol<T>::value && !is_time_value<T>::value && !is_color<T>::value && is_class<T>::value, int>::type = 0>
    atoms to_atoms(const T& container) {
        atoms  as(container.size());
        size_t index = 0;

        for (const auto& item : container) {
            as[index] = item;
            ++index;
        }
        return as;
    }


    /// Copy values from a color to a vector of atoms of size=4.
    /// @tparam	T	The type of the input value.
    /// @param	v	The value to be copied.
    /// @return		A vector of atoms

    template<class T, typename enable_if<is_color<T>::value, int>::type = 0>
    atoms to_atoms(const T& v) {
        atoms as{v.red(), v.green(), v.blue(), v.alpha()};
        return as;
    }


    /// Copy values from any simple type to a vector of atoms of size=1.
    /// @tparam	T	The type of the input value.
    /// @param	v	The value to be copied.
    /// @return		A vector of atoms

    template<class T, typename enable_if<is_symbol<T>::value || is_time_value<T>::value || !is_class<T>::value, int>::type = 0>
    atoms to_atoms(const T& v) {
        atoms as{v};
        return as;
    }


    /// Copy values out from a vector of atoms to the desired container class
    /// @tparam	T	The type of the container
    /// @param	as	The vector atoms containing the desired data
    /// @return		The container of the values

    template<class T, typename enable_if<!is_symbol<T>::value && !is_time_value<T>::value && !is_color<T>::value && is_class<T>::value, int>::type = 0>
    T from_atoms(const atoms& as) {
        T container;

        container.reserve(as.size());
        for (const auto& a : as)
            container.push_back(a);
        return container;
    }


    /// Copy values out from a vector of atoms to the desired color type
    /// @tparam	T	The type of the destination (a ui::color)
    /// @param	as	The vector atoms containing the desired data
    /// @return		The color

    template<class T, typename enable_if<is_color<T>::value, int>::type = 0>
    T from_atoms(const atoms& as) {
        ui::color c{as[0], as[1], as[2], as[3]};    // TODO: bounds-checking
        return c;
    }


    /// Copy a value out from a vector of atoms to the desired type
    /// @tparam	T	The type of the destination variable
    /// @param	as	The vector atoms containing the desired data
    /// @return		The value

    template<class T, typename enable_if<!std::is_enum<T>::value && (is_symbol<T>::value || is_time_value<T>::value || !is_class<T>::value), int>::type = 0>
    T from_atoms(const atoms& as) {
        return static_cast<T>(as[0]);
    }


#ifndef C74_MIN_NO_ENUM_CHECKS

    /// Copies a value out from a vector of atoms to the desired type -- an enum
    /// The value is restricted to the range of the enum
    /// The enum *must* follow the convention of starting a zero, incrementing sequentially, and ending with 'enum_count'
    /// If this convention is untenable then define C74_MIN_NO_ENUM_CHECKS in your preprocessor symbols.
    ///
    /// @tparam	T	The (enum) type of the data
    /// @param	as	The vector atoms containing the desired data
    /// @return		The enum value

    template<class T, typename enable_if<std::is_enum<T>::value, int>::type = 0>
    T from_atoms(const atoms& as) {
        auto index = static_cast<long>(as[0]);
        auto size  = static_cast<long>(T::enum_count);

        if (index < 0)
            index = 0;
        else if (index >= size)
            index = size - 1;
        return T(index);
    }

#else

    // A version of the above but which is not safe (can't check for out-of-bounds values)

    template<class T, typename enable_if<std::is_enum<T>::value, int>::type = 0>
    T from_atoms(const atoms& as) {
        auto index = static_cast<long>(as[0]);
        return T(index);
    }

#endif    // C74_MIN_NO_ENUM_CHECKS
}        // namespace c74::min
