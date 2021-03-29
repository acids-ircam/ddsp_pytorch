/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_mock.h"






namespace c74 {
namespace max {
    struct t_qelem;

    MOCK_EXPORT t_qelem* qelem_new(void* obj, method fn) {
        return nullptr;
    }

    MOCK_EXPORT void qelem_free(t_qelem* qelem) {
        ;
    }

    MOCK_EXPORT void qelem_set(t_qelem* q) {
        ;
    }


    MOCK_EXPORT short systhread_ismainthread(void) {
        return true;
    }

    MOCK_EXPORT short systhread_istimerthread(void) {
        return false;
    }

    MOCK_EXPORT void* defer(void* ob,method fn,t_symbol* sym,short argc,t_atom* argv) {
        return nullptr;
    }


    MOCK_EXPORT float sys_getsr(void) {
        return 44100;
    }

    MOCK_EXPORT int sys_getblksize(void) {
        return 32;
    }


    MOCK_EXPORT void binbuf_insert(void *x, t_symbol *s, short argc, t_atom *argv) {}


    MOCK_EXPORT short locatefile_extended(char* name, short* outvol, t_fourcc* outtype, const t_fourcc* filetypelist, short numtypes) {
        return 0;
    }

    MOCK_EXPORT void typelist_make(t_fourcc *types, long include, short *numtypes) {}


    MOCK_EXPORT short path_getpath(short path, const char *name, short *outpath) {
        return 0;
    }


    MOCK_EXPORT short path_createfolder(const short path, const char* name, short* newpath) {
        return 0;
    }

    MOCK_EXPORT short path_getmoddate(short path, t_ptr_uint* date) {
        return 0;
    }

    MOCK_EXPORT short path_getfilemoddate(const char* filename, const short path, t_ptr_uint* date) {
        *date = 0;
        return 0;
    }

    MOCK_EXPORT short path_nameconform(const char* src, char* dst, long style, long type) {
        return 0;
    }

    MOCK_EXPORT short path_frompathname(const char* name, short* path, char* filename) {
        return 0;
    }

    MOCK_EXPORT t_max_err path_toabsolutesystempath(const short in_path, const char* in_filename, char* out_filepath) {
        return 0;
    }

    MOCK_EXPORT t_max_err object_attr_touch(t_object* x, t_symbol* attrname) {
        return 0;
    }

    MOCK_EXPORT t_max_err object_attr_getvalueof(t_object* x, t_symbol* s, long* argc, t_atom** argv) {
        return 0;
    }

    MOCK_EXPORT t_max_err object_attr_setvalueof(t_object* x, t_symbol* s, long argc, const t_atom* argv) {
        return 0;
    }

    MOCK_EXPORT t_atom_long object_attr_getlong(void* x, t_symbol* s) {
        return 0;
    }


    MOCK_EXPORT t_object* attr_offset_new(const char* name, const t_symbol* type, long flags, const method mget, const method mset, long offset) {
        return nullptr;
    }

    MOCK_EXPORT t_object* attr_offset_array_new(const char* name, t_symbol* type, long size, long flags, method mget, method mset, long offset_count, long offset) {
        return nullptr;
    }



    MOCK_EXPORT void attr_args_process(void* x, const short ac, const t_atom* av) {}

    MOCK_EXPORT long attr_args_offset(const short ac, const t_atom* av) {
        return 0;
    }


    MOCK_EXPORT void attr_dictionary_process(void* x, t_dictionary* d) {}

    MOCK_EXPORT void attr_dictionary_check(void* x, t_dictionary* d) {}

    MOCK_EXPORT t_object* attribute_new_parse(const char* attrname, t_symbol* type, long flags, const char* parsestr) {
        return nullptr;
    }

    MOCK_EXPORT t_max_err class_sticky(t_class* x, t_symbol* stickyname, t_symbol* s, t_object* o) {
        return 0;
    }

    MOCK_EXPORT t_max_err class_sticky_clear(t_class* x, t_symbol* stickyname, t_symbol* s) {
        return 0;
    }

    MOCK_EXPORT t_dictionary* object_dictionaryarg(const long ac, const t_atom* av) {
        return nullptr;
    }



    MOCK_EXPORT method object_getmethod(void* x, t_symbol* s) {
        return nullptr;
    }

    MOCK_EXPORT t_symbol* symbol_unique(void) {
        return nullptr;
    }


    MOCK_EXPORT t_max_err object_attach_byptr_register(void* x, void* object_to_attach, const t_symbol* reg_name_space) {
        return 0;
    }


    MOCK_EXPORT t_max_err object_retain(t_object*) {
        return 0;
    }


    MOCK_EXPORT t_object* object_new_imp(void* p1, void* p2, void* p3, void* p4, void* p5, void* p6, void* p7, void* p8, void* p9, void* p10) {
        return nullptr;
    }


    using t_jit_object = t_object;
    using t_jit_err = long;

    MOCK_EXPORT t_jit_object* jit_object_new_imp(void* classname, void* p1, void* p2, void* p3, void* p4, void* p5, void* p6, void* p7, void* p8, void* dummy) {
        return object_new_imp(classname, p1, p2, p3, p4, p5, p6, p7, p8, dummy);
    }


    MOCK_EXPORT t_jit_err jit_class_addadornment(void* c, t_jit_object* o) {
        return 0;
    }


    MOCK_EXPORT t_jit_err jit_class_addmethod(void* c, method m, const char* name, ...) {
        return 0;
    }


    MOCK_EXPORT t_jit_err jit_class_register(void* c) {
        return 0;
    }


    MOCK_EXPORT t_jit_err jit_class_addattr(void* c, t_jit_object* attr) {
        return 0;
    }


    MOCK_EXPORT t_max_err object_addattr_parse(t_object* x, const char* attrname, t_symbol* type, long flags, const char* parsestr) {
        return 0;
    }


    MOCK_EXPORT t_max_err class_attr_addattr_parse(t_class* c, const char* attrname, const char* attrname2, t_symbol* type, long flags, const char* parsestr) {
        return 0;
    }

    MOCK_EXPORT t_max_err class_attr_addattr_format(t_class* c, const char* attrname, const char* attrname2, const t_symbol* type, long flags, const char* fmt, ...) {
        return 0;
    }


    MOCK_EXPORT t_max_err class_attr_addattr_atoms(t_class* c, const char* attrname, const char* attrname2, t_symbol* type, long flags, long ac, t_atom* av) {
        return 0;
    }

    MOCK_EXPORT t_max_err class_attr_get(t_class* x, t_symbol* attrname) {
        return 0;
    }

    MOCK_EXPORT void class_time_addattr(t_class *c, char *attrname, char *attrlabel, long flags) {
        return;
    }


    MOCK_EXPORT t_max_err class_addtypedwrapper(t_class *x, method m, const char *name, ...) {
        return 0;
    }

    MOCK_EXPORT t_max_err class_parameter_register_default_color(t_class* c, t_symbol* attrname, t_symbol* colorname) {
        return 0;
    }

    MOCK_EXPORT t_max_err object_parameter_color_get(t_object* x, t_symbol* s, t_jrgba* rgba) { 
        return 0;
    }



    using t_jbox = t_object;


    MOCK_EXPORT t_max_err jbox_new(t_jbox* b, long flags, long argc, t_atom* argv) { return 0; }
    MOCK_EXPORT void jbox_free(t_jbox *b) {}
    MOCK_EXPORT void jbox_ready(t_jbox* b) {}
    MOCK_EXPORT void jbox_initclass(t_jbox* b) {}
    MOCK_EXPORT void jbox_redraw(t_jbox* b) {}


    using t_jgraphics = t_object;
    using t_jgraphics_format = int;
    using t_jgraphics_line_join = int;
    using t_jgraphics_line_cap = int;
    struct t_jrgba;
    struct t_jsurface;

    MOCK_EXPORT void jgraphics_set_line_cap(t_jgraphics* g, t_jgraphics_line_cap line_cap) {}
    MOCK_EXPORT void jgraphics_set_line_join(t_jgraphics* g, t_jgraphics_line_join line_join) {}
    MOCK_EXPORT void jgraphics_set_line_width(t_jgraphics* g, double width) {}
    MOCK_EXPORT void jgraphics_line_to(t_jgraphics* g, double x, double y) {}
    MOCK_EXPORT void jgraphics_line_draw_fast(t_jgraphics* g, double x1, double y1, double x2, double y2, double linewidth) {}
    MOCK_EXPORT void jgraphics_move_to(t_jgraphics* g, double x, double y) {}
    MOCK_EXPORT void jgraphics_rectangle_rounded(t_jgraphics* g, double x, double y, double width, double height, double ovalwidth, double ovalheight) {}
    MOCK_EXPORT void jgraphics_ellipse(t_jgraphics* g, double x, double y, double width, double height) {}
    MOCK_EXPORT void jgraphics_rectangle_fill_fast(t_jgraphics* g, double x, double y, double width, double height) {}
    MOCK_EXPORT void jgraphics_fill(t_jgraphics* g) {}
    MOCK_EXPORT void jgraphics_stroke(t_jgraphics* g) {}
    MOCK_EXPORT void jgraphics_set_source_jrgba(t_jgraphics* g, t_jrgba* rgba) {}
    MOCK_EXPORT t_jsurface* jgraphics_image_surface_create(t_jgraphics_format format, int width, int height) {
		return nullptr;
    }
    MOCK_EXPORT void jgraphics_surface_destroy(const t_jsurface* surface) {}

     MOCK_EXPORT unsigned char* jgraphics_image_surface_lockpixels(
		t_jsurface* s, int x, int y, int width, int height, int* linestride, int* pixelstride) {
		return nullptr;
	}

	MOCK_EXPORT void jgraphics_image_surface_unlockpixels(t_jsurface* s, unsigned char* data) {}

	MOCK_EXPORT void jgraphics_image_surface_draw(t_jgraphics* g, t_jsurface* s, t_rect srcRect, t_rect destRect) { }

    MOCK_EXPORT void jgraphics_image_surface_clear(t_jsurface* s, int x, int y, int width, int height) { }

    MOCK_EXPORT int jgraphics_image_surface_get_width(t_jsurface* s) { return 0; }
    MOCK_EXPORT int jgraphics_image_surface_get_height(t_jsurface* s) { return 0; }

    MOCK_EXPORT t_jgraphics* jgraphics_create(t_jsurface* target) { return nullptr; }
    MOCK_EXPORT void jgraphics_destroy(t_jgraphics* g) { }

    MOCK_EXPORT t_max_err jbox_invalidate_layer(t_object* b, t_object* view, t_symbol* name) { return 0; }
    MOCK_EXPORT t_jgraphics* jbox_start_layer(t_object* b, t_object* view, t_symbol* name, double width, double height) { return nullptr; }
    MOCK_EXPORT t_max_err jbox_end_layer(t_object* b, t_object* view, t_symbol* name) { return 0; }
    MOCK_EXPORT t_max_err jbox_paint_layer(t_object* b, t_object* view, t_symbol* name, double x, double y) { return 0; }

    MOCK_EXPORT void max_jit_obex_gimmeback_dumpout(void *x, t_symbol *s, long ac, t_atom *av) {
        return;
    }



    MOCK_EXPORT void *max_jit_object_alloc(t_class *mclass, t_symbol *jitter_classname) {
        return nullptr;
    }

    MOCK_EXPORT void max_jit_object_free(void *x) {}
    MOCK_EXPORT void max_jit_class_obex_setup(t_class *mclass, long oboffset) {}
    MOCK_EXPORT t_jit_err max_jit_class_addattr(t_class *mclass, void *attr) {
        return 0;
    }
    MOCK_EXPORT void max_jit_class_wrap_standard(t_class *mclass, t_class *jclass, long flags) {}
    MOCK_EXPORT void max_jit_class_wrap_addmethods(t_class *mclass, t_class *jclass) {}
    MOCK_EXPORT void max_jit_class_wrap_addmethods_flags(t_class *mclass, t_class *jclass, long flags) {}
    MOCK_EXPORT void max_jit_class_wrap_attrlist2methods(t_class *mclass, t_class *jclass) {}
    MOCK_EXPORT void max_jit_class_addmethod_defer(t_class *mclass, method m, const char *s) {}
    MOCK_EXPORT void max_jit_class_addmethod_defer_low(t_class *mclass, method m, const char *s) {}
    MOCK_EXPORT void max_jit_class_addmethod_usurp(t_class *mclass, method m, const char *s) {}
    MOCK_EXPORT void max_jit_class_addmethod_usurp_low(t_class *mclass, method m, const char *s) {}


    MOCK_EXPORT t_jit_err max_jit_class_mop_wrap(t_class* mclass, t_class* jclass, long flags) {
        return 0;
    }

    MOCK_EXPORT t_jit_err max_jit_mop_setup_simple(t_object*, t_object* o, long argc, t_atom* argv) {
        return 0;
    }

    MOCK_EXPORT t_jit_err max_jit_mop_notify(void *x, t_symbol *s, t_symbol *msg) { return 0; }
    MOCK_EXPORT void max_jit_mop_assist(t_object*) {} // note: prototype wrong
    MOCK_EXPORT int max_jit_mop_getoutputmode(t_object*) { return 0; }
    MOCK_EXPORT void max_jit_mop_outputmatrix(t_object*) {}
    MOCK_EXPORT void max_jit_mop_free(t_object*) {}
    MOCK_EXPORT void* max_jit_obex_jitob_get(t_object*) { return nullptr; }
    MOCK_EXPORT void* max_jit_obex_adornment_get(t_object* self, t_symbol* name) { return nullptr; }
    MOCK_EXPORT void max_jit_attr_args(t_object*, long argc, t_atom* argv) {}
    MOCK_EXPORT void* jit_object_alloc(t_class* c) { return object_alloc(c); }
    MOCK_EXPORT void jit_object_free(t_object* x) { object_free(x); }



#ifdef __APPLE__
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-variable"
#endif
    static const int JIT_MATRIX_MAX_DIMCOUNT = 32; 			///< maximum dimension count @ingroup jitter
    static const int JIT_MATRIX_MAX_PLANECOUNT = 32; 			///< maximum plane count @ingroup jitter
#ifdef __APPLE__
    #pragma clang diagnostic pop
#endif

    struct t_jit_matrix_info {
        long		size = 0;			///< in bytes (0xFFFFFFFF=UNKNOWN)
        t_symbol*	type = nullptr;			///< primitifve type (char, long, float32, or float64)
        long		flags = 0;			///< flags to specify data reference, handle, or tightly packed
        long		dim_count = 0;		///< number of dimensions
        long		dim[JIT_MATRIX_MAX_DIMCOUNT];		///< dimension sizes
        long		dimstride[JIT_MATRIX_MAX_DIMCOUNT]; ///< stride across dimensions in bytes
        long		plane_count = 0;		///< number of planes
    };

    MOCK_EXPORT void jit_parallel_ndim_simplecalc1(method fn, void *data, long dim_count, long *dim, long plane_count, t_jit_matrix_info *minfo1, char *bp1, long flags1)
    {}
    MOCK_EXPORT void jit_parallel_ndim_simplecalc2(method fn, void *data, long dim_count, long *dim, long plane_count, t_jit_matrix_info *minfo1, char *bp1,
        t_jit_matrix_info *minfo2, char *bp2, long flags1, long flags2)
    {}

    #ifndef C74_X64
    MOCK_EXPORT void* jit_object_new(t_symbol* name) { return jit_object_new_imp(name, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr); }
    #endif

}}

