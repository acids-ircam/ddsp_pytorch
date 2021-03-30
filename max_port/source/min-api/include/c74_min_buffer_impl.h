/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_min_buffer.h"

namespace c74::min {


    template<>
    buffer_lock<true>::buffer_lock(buffer_reference& a_buffer_ref)
    : m_buffer_ref { a_buffer_ref } {
        m_buffer_obj = buffer_ref_getobject(m_buffer_ref.m_instance);
        m_tab        = buffer_locksamples(m_buffer_obj);
        // TODO: handle case where tab is null -- can't throw an exception in audio code...
    }

    template<>
    buffer_lock<false>::buffer_lock(buffer_reference& a_buffer_ref)
    : m_buffer_ref { a_buffer_ref } {
        max::t_buffer_info info;

        m_buffer_obj = buffer_ref_getobject(m_buffer_ref.m_instance);
        buffer_edit_begin(m_buffer_obj);
        buffer_getinfo(m_buffer_obj, &info);
        m_tab = info.b_samples;
    }


    template<>
    buffer_lock<true>::~buffer_lock() {
        if (m_tab)
            buffer_unlocksamples(m_buffer_obj);
    }

    template<>
    buffer_lock<false>::~buffer_lock() {
        buffer_edit_end(m_buffer_obj, true);
    }

}    // namespace c74::min
