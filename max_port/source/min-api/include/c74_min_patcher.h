/// @file
///	@ingroup 	minapi
///	@copyright  Copyright 2020 The Min-API Authors. All rights reserved.
///	@license           Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {


    /// An instance of an object.
    /// This could be a box, a patcher, or anything else that is a live instance of a class in Max.

    class instance {
        struct messinfo {
            max::t_object*  ob;
            max::method     fn;
            int             type;
        };


    public:
        instance(max::t_object* an_instance = nullptr)
        : m_instance { an_instance }
        {}

        virtual ~instance() {
            if (m_own && m_instance)
                object_free(m_instance);
        }


        operator max::t_object*() const {
            return m_instance;
        }

        operator bool() const {
            return m_instance != nullptr;
        }


        template<typename T1>
        void instantiate(const symbol a_name, const T1 arg1) {
            if (m_instance && m_own)
                max::object_free(m_instance);
            m_instance = max::object_new(max::CLASS_NOBOX, a_name, arg1, 0);
        }


        /// call a method on an instance

        atom operator()(const symbol method_name) {
            auto m { find_method(method_name) };

            if (m.type == max::A_GIMME) {
                atoms as {};
                return max::object_method_typed(m.ob, method_name, 0, nullptr, nullptr);
            }
            else if (m.type == max::A_GIMMEBACK) {
                atoms       as {};
                max::t_atom rv {};

                max::object_method_typed(m.ob, method_name, 0, nullptr, &rv);
                return rv;
            }
            else {
                return m.fn(m.ob);
            }
        }


        template<typename T1>
        atom operator()(const symbol method_name, const T1 arg1) {
            auto m { find_method(method_name) };

            if (m.type == max::A_GIMME) {
                atoms   as { arg1 };
                return max::object_method_typed(m.ob, method_name, static_cast<long>(as.size()), &as[0], nullptr);
            }
            else if (m.type == max::A_GIMMEBACK) {
                atoms       as { arg1 };
                max::t_atom rv {};

                max::object_method_typed(m.ob, method_name, static_cast<long>(as.size()), &as[0], &rv);
                return rv;
            }
            else {
                if (typeid(T1) != typeid(atom))
                    return m.fn(m.ob, arg1);
                else {
                    // atoms must be converted to native types and then reinterpreted as void*
                    // doubles cannot be converted -- supporting those will need to be handled separately
                    return m.fn(m.ob, atom_to_generic(arg1));
                }
            }
        }


        template<typename T1, typename T2>
        atom operator()(const symbol method_name, const T1 arg1, const T2 arg2) {
            auto m { find_method(method_name) };

            if (m.type == max::A_GIMME) {
                atoms   as { arg1, arg2 };
				return max::object_method_typed(m.ob, method_name, static_cast<long>(as.size()), &as[0], nullptr);
            }
            else if (m.type == max::A_GIMMEBACK) {
                atoms       as { arg1, arg2 };
                max::t_atom rv {};

                max::object_method_typed(m.ob, method_name, static_cast<long>(as.size()), &as[0], &rv);
                return rv;
            }
            else {
                if (typeid(T1) != typeid(atom))
                    return m.fn(m.ob, arg1, arg2);
                else {
                    // atoms must be converted to native types and then reinterpreted as void*
                    // doubles cannot be converted -- supporting those will need to be handled separately
                    return m.fn(m.ob, atom_to_generic(arg1), atom_to_generic(arg2));
                }
            }
        }


        /// Set and get attributes of an instance

        void set(const symbol attribute_name, const symbol value) {
            max::object_attr_setsym(m_instance, attribute_name, value);
        }

        void set(const symbol attribute_name, const char value) {
            max::object_attr_setchar(m_instance, attribute_name, value);
        }

        template<typename T>
        T get(const symbol attribute_name) const {
            long argc {};
            max::t_atom* argv {};

            max::object_attr_getvalueof(m_instance, attribute_name, &argc, &argv);
            return static_cast<T>(atom(argv));
        }


    protected:
        max::t_object*  m_instance;
        bool            m_own {};

        auto find_method(const symbol a_method_name) -> messinfo {
            max::t_object* ob = m_instance;

            for (max::t_messlist* mess = ob->o_messlist; max::t_symbol* s = mess->m_sym; ++mess) {
                if (a_method_name == symbol(mess->m_sym)) {
                    return {ob, mess->m_fun, mess->m_type[0] };
                }
            }

            // no message found for the box, so call the object...
            // TODO: check to see if it is a box before assuming it is...
            ob = max::jbox_get_object(m_instance);

            for (max::t_messlist* mess = ob->o_messlist; max::t_symbol* s = mess->m_sym; ++mess) {
                if (a_method_name == symbol(mess->m_sym)) {
                     return {ob, mess->m_fun, mess->m_type[0] };
                }
            }

            // not found
            return { nullptr, nullptr, 0 };
        }


    private:
        void* atom_to_generic(const atom& a) {
            if (a.type() == message_type::int_argument)
                return reinterpret_cast<void*>( static_cast<max::t_atom_long>(a) );
            // else if (a.type() == message_type::float_argument)
            //  return reinterpret_cast<void*>( static_cast<max::t_atom_float>(a) );
            else if (a.type() == message_type::symbol_argument)
                return reinterpret_cast<void*>( static_cast<max::t_symbol*>(a) );
            else
                return reinterpret_cast<void*>( static_cast<max::t_object*>(a) );
        }

    };


    class box : public instance {
    public:
        box(max::t_object *a_box)
        : instance { a_box }
        {}

        symbol classname() const {
            return max::jbox_get_maxclass(m_instance);
        }
        
        symbol path() const {
            return max::jbox_get_boxpath(m_instance);
        }

        symbol name() const {
            return max::jbox_get_varname(m_instance);
        }

        void name(const symbol a_new_scripting_name) {
            max::jbox_set_varname(m_instance, a_new_scripting_name);
        }
    };
    
    using boxes = std::vector<box>;


    class device : public instance {
    public:
        device(max::t_object* a_device = nullptr)
        : instance { a_device }
        {}
    };


    /// A reference to a buffer~ object.
    /// The buffer_reference automatically adds the management hooks required for your object to work with a buffer~.
    /// This includes adding a 'set' message and a 'dblclick' message as well as dealing with notifications and binding.
    /// @ingroup buffers

    class patcher : public instance {
    public:
        /// Create a reference to a buffer~ object.
        /// @param	an_owner	The owning object for the buffer reference. Typically you will pass `this`.
        /// @param	a_function	An optional function to be executed when the buffer reference issues notifications.
        ///						Typically the function is defined using a C++ lambda with the #MIN_FUNCTION signature.

        // takes a single arg, but cannot be marked explicit unless we are willing to decorate all using code with a cast to this type
        // thus we ignore the advice of C.46 @ https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md

        patcher(c74::max::t_object* a_patcher)
        : instance { a_patcher }
        {}


        min::device device() const {
            max::t_object* m4l_device {};
            max::object_obex_lookup(m_instance, symbol("##plugdevice##"), &m4l_device);
            return m4l_device;
        }


        /// Bind the buffer reference to a buffer with a different name.
        /// @param	name	The name of the buffer~ with which to bind the reference.

        min::boxes boxes() {
            m_boxes.clear();

            auto box = max::jpatcher_get_firstobject(m_instance);
            while (box) {
                m_boxes.push_back(box);
                box = max::jbox_get_nextobject(box);
            }
            return m_boxes;
        }
        
        symbol name() const {
             return max::jpatcher_get_name(m_instance);
        }

    private:
        min::boxes      m_boxes     {};
    };


}    // namespace c74::min
