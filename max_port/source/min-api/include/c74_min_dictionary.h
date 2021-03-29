/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {

    class dict {
    public:
        /// Create (or reference an existing) dictionary by name
        dict(const symbol name) {
            auto d = max::dictobj_findregistered_retain(name);

            if (!d) {    // didn't find a dictionary with that name, so create it
                d                 = max::dictionary_new();
                max::t_symbol* s  = name;
                m_instance        = max::dictobj_register(d, &s);
            }
            else {
                m_instance = d;
            }
        }

        /// Create an unregistered dictionary from dict-syntax
        dict(const atoms& content) {
            max::dictobj_dictionaryfromatoms(&m_instance, static_cast<long>(content.size()), &content[0]);
        }

        /// Create an unregistered dictionary
        /// @param d				optionally getting a handle to an old-school t_dictionary
        /// @param take_ownership	defaults to true, change to false only in exceptional cases
        dict(max::t_dictionary* d = nullptr, const bool take_ownership = true) {
            if (d == nullptr)
                m_instance = max::dictionary_new();
            else {
                if (take_ownership)
                    max::object_retain(d);
                else
                    m_has_ownership = false;
                m_instance = d;
            }
        }


        dict(const atom an_atom_containing_a_dict) {
            auto a     = static_cast<const max::t_atom*>(&an_atom_containing_a_dict);
            m_instance = static_cast<max::t_dictionary*>(max::atom_getobj(a));
            if (!m_instance)
                error("no dictionary in atom");
            auto err = max::object_retain(m_instance);
            error(err, "failed to retain dictionary instance");
        }


        ~dict() {
            if (m_has_ownership)
                object_free(m_instance);
        }


        dict& operator=(const dict& value) {
            max::dictionary_clone_to_existing(value.m_instance, m_instance);
            return *this;
        }


        dict& operator=(const atom& value) {
            auto a = static_cast<const max::t_atom*>(&value);
            if (max::atomisdictionary(a))
                m_instance = static_cast<max::t_dictionary*>(max::atom_getobj(a));
            return *this;
        }


        /// Cast the dictionary to a Max t_object so that, e.g., it can be used with the C Max API.
        /// Be exceedingly careful with this!
        /// When you pass the pointer out are you pointing the instance without incrementing the reference count?

        // TODO: We need copy (retain) and move (don't retain) semantics ????
        // TODO: we don't have a copy constructor!

        operator max::t_object*() const {
            max::object_retain(m_instance);
            return static_cast<max::t_object*>(m_instance);
        }


        // bounds check: if key doesn't exist, throw
        atom_reference at(const symbol key) {
            long         argc = 0;
            max::t_atom* argv = nullptr;
            auto         err  = max::dictionary_getatoms(m_instance, key, &argc, &argv);

            error(err, "could not get key from dictionary");
            return atom_reference(argc, argv);
        }


        // bounds check: if key doesn't exist, create it
        atom_reference operator[](symbol key) {
            if (!max::dictionary_hasentry(m_instance, key))
                max::dictionary_appendatom(m_instance, key, &atoms{0}[0]);
            return at(key);
        };

        atom_reference operator[](int key) {
            symbol skey{key};
            return (*this)[skey];
        };


        symbol name() const {
            return dictobj_namefromptr(m_instance);
        }


        bool valid() const {
            return m_instance != nullptr;
        }


        void clear() {
            dictionary_clear(m_instance);
        }


        void copyunique(const dict& source) {
            dictionary_copyunique(m_instance, source.m_instance);
        }


        void touch() {
            object_notify(m_instance, k_sym_modified, nullptr);
        }


    private:
        max::t_dictionary* m_instance       { nullptr };
        bool               m_has_ownership  { true };
    };

}    // namespace c74::min
