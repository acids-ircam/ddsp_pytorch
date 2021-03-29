/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {


    using t_jbox = t_object;
    using t_pxjbox = t_jbox;
    
    struct t_rect {
		double x;         ///< The horizontal origin
		double y;         ///< The vertical origin
		double width;     ///< The width
		double height;    ///< The height
	};

    struct t_jrgba {
		double red;      ///< Red component in the range [0.0, 1.0]
		double green;    ///< Green component in the range [0.0, 1.0]
		double blue;
		double alpha;    ///< Alpha (transparency) component in the range [0.0, 1.0]
	};


    MOCK_EXPORT void class_dspinitjbox(t_class* c) {}

    MOCK_EXPORT void class_attr_setstyle(t_class *c, const char *name) {}

    MOCK_EXPORT void z_jbox_dsp_setup(t_pxjbox* x, long nsignals) {}

    MOCK_EXPORT void z_jbox_dsp_free(t_pxjbox* x) {}

    MOCK_EXPORT t_max_err jbox_notify(t_jbox* b, t_symbol* s, t_symbol* msg, void* sender, void* data) {
        return 0;
    }

    MOCK_EXPORT t_max_err jbox_get_rect_for_view(t_object* box, t_object* patcherview, t_rect* rect) {
		return 0;
    }

    MOCK_EXPORT t_object* patcherview_get_jgraphics(t_object* pv) {
		return nullptr;
    }



}}
