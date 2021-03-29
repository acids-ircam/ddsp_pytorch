/// @file
///    @ingroup     minapi
///    @copyright    Copyright 2018 The Min-API Authors. All rights reserved.
///    @license    Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {

    using t_dictionary = t_object;

    MOCK_EXPORT t_dictionary* dictobj_findregistered_retain(t_symbol* name) {
        return nullptr;
    }

    MOCK_EXPORT t_max_err dictobj_release(t_dictionary* d) {
        return 0;
    }

    MOCK_EXPORT t_dictionary* dictionary_new(void) {
        return nullptr;
    }

    MOCK_EXPORT t_max_err dictionary_appendatomarray(t_dictionary* d, t_symbol* key, t_object* value) {
        return 0;
    }

    MOCK_EXPORT t_max_err dictionary_appenddictionary(t_dictionary* d, t_symbol* key, t_object* value) {
        return 0;
    }

    MOCK_EXPORT t_max_err dictionary_appendlong(t_dictionary* d, t_symbol* key, t_atom_long value) {
        return 0;
    }

    MOCK_EXPORT t_dictionary* dictobj_register(t_dictionary* d, t_symbol** name) {
        return nullptr;
    }


    MOCK_EXPORT t_max_err dictionary_clone_to_existing(const t_dictionary* d, t_dictionary* dc) {
        return 0;
    }

    MOCK_EXPORT t_max_err dictionary_copyunique(t_dictionary* d, t_dictionary* copyfrom) {
        return 0;
    }


    MOCK_EXPORT t_symbol* dictobj_namefromptr(t_dictionary* d) {
        return nullptr;
    }

    MOCK_EXPORT t_max_err dictobj_dictionaryfromatoms(t_dictionary** d, const long argc, const t_atom* argv) {
        return 0;
    }

    MOCK_EXPORT t_max_err object_notify(void* x, const t_symbol* s, void* data) {
        return 0;
    }



}} // namespace c74::max
