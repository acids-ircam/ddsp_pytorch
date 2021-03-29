/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {

    /// A callback function used to handle an argument from the object box at instantiation.
    /// Typically this is provided to argument as a lamba function using the #MIN_ARGUMENT_FUNCTION macro.
    /// @param	a	An atom which is the value provided as the argument to be handled.
    /// @see		MIN_ARGUMENT_FUNCTION
    /// @see		MIN_FUNCTION

    using argument_function = std::function<void(const atom& a)>;


    /// Provide the correct lamba function prototype for the min::argument constructor.
    /// @see argument
    /// @see argument_function

    #define MIN_ARGUMENT_FUNCTION [this](const c74::min::atom& arg)


    // Represents any type of argument declaration.
    // Used internally to allow heterogenous containers of argument declarations.

    class argument_base {
    public:
        argument_base(object_base* an_owner, const std::string& a_name, const description& a_description, const bool required, const argument_function& a_function)
        : m_owner       { an_owner }
        , m_name        { a_name }
        , m_description { a_description }
        , m_required    { required }
        , m_function    { a_function } {
            m_owner->register_argument(this);
        }


        /// Execute this argument's handler if one was provided.
        /// This is called internally when the Max object instance is created.
        /// @param arg	The value for the argument.

        void operator()(const atom& arg) {
            if (m_function)
                m_function(arg);
        }


        /// Return the human-readable documentation of the argument.
        /// @return	The description string documenting the argument.

        std::string description_string() const {
            return m_description;
        }


        /// Return the name of the argument.
        /// This is used both to generate documentation and thus object box auto-completion.
        /// @return	The name of the argument.

        symbol name() const {
            return m_name;
        }


        /// Does this argument require a value to be provided by the user? Or is it optional?
        /// @return True if this argument is required. False if it is optional.

        bool required() const {
            return m_required;
        }


        // Declaring this as pure virtual because, yes we need this defined for all argument decls,
        // but also to enforce that no one tries to instantiate an argument_base directly.

        virtual std::string type() const = 0;

    protected:
        object_base*            m_owner;
        const symbol            m_name;
        const description       m_description;
        const bool              m_required;
        const argument_function m_function;
    };


    /// An argument declaration.
    /// The argument declaration class is a templated type where the type is the type of argument
    /// a user may enter into the object box in Max.
    ///
    /// An argument declaration may be either active or passive.
    /// An active argument declaration will be executed when an instance of your Min class is created.
    /// An argument declaration will be passive, meaning it only provides documentation, if your
    /// class defines a custom constructor.
    /// In the later case you must process the arguments manually in your constructor.
    ///
    /// @tparam	T The type of argument a user may type into the object box in Max.

    template<class T>
    class argument : public argument_base {
    public:
        /// Creates an argument declaration for your class.
        /// @param	an_owner		The Min object instance that owns this argument. Typically you should pass 'this'.
        /// @param	a_name			A string specifying a symbolic name for this argument.
        /// @param	a_description	Documentation string for this argument.
        /// @param	a_function		Optional function to be called when the argument is processed at object instantiation.

        argument(object_base* an_owner, const std::string& a_name, const description& a_description, const argument_function& a_function = {})
        : argument_base(an_owner, a_name, a_description, false, a_function)
        {}


        /// Creates an argument declaration for your class.
        /// @param	an_owner		The Min object instance that owns this argument. Typically you should pass 'this'.
        /// @param	a_name			A string specifying a symbolic name for this argument.
        /// @param	a_description	Documentation string for this argument.
        /// @param	required		If true the argument _must_ be provided by the user. Otherwise the argument is optional.
        /// @param	a_function		Optional function to be called when the argument is processed at object instantiation.

        argument(object_base* an_owner, const std::string& a_name, const description& a_description, const bool required, const argument_function& a_function = {})
        : argument_base(an_owner, a_name, a_description, required, a_function)
        {}


        /// Return the type of the argument as a string.
        /// This is used to generate documentation and thus object box auto-completion.
        /// @return	The type of the argument.

        std::string type() const override {
            if (is_same<T, bool>::value)
                return "bool";
            else if (is_same<T, number>::value)
                return "number";
            else if (is_same<T, float>::value)
                return "float";
            else if (is_same<T, double>::value)
                return "float";
            else if (is_same<T, int>::value)
                return "int";
            else if (is_same<T, long>::value)
                return "int";
            else if (is_same<T, symbol>::value)
                return "symbol";
            else
                return "";    // includes 'anything' type
        }
    };

}    // namespace c74::min
