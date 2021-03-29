/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#define C74_NO_DEPRECATION
#include "c74_max.h"

namespace c74 {
namespace max {

    // jit.error.h

    typedef t_atom_long t_jit_err;

    static const int JIT_ERR_NONE 				= 0          ;
    static const int JIT_ERR_GENERIC			= 	('EROR') ;
    static const int JIT_ERR_INVALID_OBJECT		= ('INOB')   ;
    static const int JIT_ERR_OBJECT_BUSY		= 	('OBSY') ;
    static const int JIT_ERR_OUT_OF_MEM			= ('OMEM')   ;
    static const int JIT_ERR_INVALID_PTR		= 	('INVP') ;
    static const int JIT_ERR_DUPLICATE			= ('DUPL')   ;
    static const int JIT_ERR_OUT_OF_BOUNDS		= ('OBND')   ;
    static const int JIT_ERR_INVALID_INPUT		= ('INVI')   ;
    static const int JIT_ERR_INVALID_OUTPUT		= ('INVO')   ;
    static const int JIT_ERR_MISMATCH_TYPE		= ('MSTP')   ;
    static const int JIT_ERR_MISMATCH_PLANE		= ('MSPL')   ;
    static const int JIT_ERR_MISMATCH_DIM		= ('MSDM')   ;
    static const int JIT_ERR_MATRIX_UNKNOWN		= ('MXUN')   ;
    static const int JIT_ERR_SUPPRESS_OUTPUT	= 	('SPRS') ;
    static const int JIT_ERR_DATA_UNAVAILABLE	= ('DUVL')   ;
    static const int JIT_ERR_HW_UNAVAILABLE		= ('HUVL')   ;


    //


    static const int MAX_JIT_MOP_FLAGS_NONE					= 0x00000000; ///< mop flag @ingroup jitter
    static const int MAX_JIT_MOP_FLAGS_OWN_ALL				= 0x0FFFFFFF; ///< mop flag @ingroup jitter
                                                                        ;
    static const int MAX_JIT_MOP_FLAGS_OWN_JIT_MATRIX		= 0x00000001; ///< mop flag @ingroup jitter
    static const int MAX_JIT_MOP_FLAGS_OWN_BANG				= 0x00000002; ///< mop flag @ingroup jitter
    static const int MAX_JIT_MOP_FLAGS_OWN_OUTPUTMATRIX		= 0x00000004; ///< mop flag @ingroup jitter
    static const int MAX_JIT_MOP_FLAGS_OWN_NAME				= 0x00000008; ///< mop flag @ingroup jitter
    static const int MAX_JIT_MOP_FLAGS_OWN_TYPE				= 0x00000010; ///< mop flag @ingroup jitter
    static const int MAX_JIT_MOP_FLAGS_OWN_DIM				= 0x00000020; ///< mop flag @ingroup jitter
    static const int MAX_JIT_MOP_FLAGS_OWN_PLANECOUNT		= 0x00000040; ///< mop flag @ingroup jitter
    static const int MAX_JIT_MOP_FLAGS_OWN_CLEAR			= 	0x00000080; ///< mop flag @ingroup jitter
    static const int MAX_JIT_MOP_FLAGS_OWN_NOTIFY			= 0x00000100; ///< mop flag @ingroup jitter
    static const int MAX_JIT_MOP_FLAGS_OWN_ADAPT			= 	0x00000200; ///< mop flag @ingroup jitter
    static const int MAX_JIT_MOP_FLAGS_OWN_OUTPUTMODE		= 0x00000400; ///< mop flag @ingroup jitter

    static const int MAX_JIT_MOP_FLAGS_ONLY_MATRIX_PROBE 	= 0x10000000; ///< mop flag @ingroup jitter

    static const int JIT_MOP_INPUT 	= 1;                                  ///< mop flag @ingroup jitter
    static const int JIT_MOP_OUTPUT	= 2;                                  ///< mop flag @ingroup jitter



    // t_jit_matrix_info flags
    static const int JIT_MATRIX_DATA_HANDLE		= 0x00000002;	///< data is handle @ingroup jitter
    static const int JIT_MATRIX_DATA_REFERENCE	= 0x00000004; 	///< data is reference to outside memory @ingroup jitter
    static const int JIT_MATRIX_DATA_PACK_TIGHT	= 0x00000008; 	///< data is tightly packed (doesn't use standard 16 byte alignment) @ingroup jitter
    static const int JIT_MATRIX_DATA_FLAGS_USE	= 0x00008000; 	/**< necessary if using handle/reference data flags when creating  @ingroup jitter
                                                     * jit_matrix, however, it is never stored in matrix */

    static const int JIT_MATRIX_MAX_DIMCOUNT		= 32; 			///< maximum dimension count @ingroup jitter
    static const int JIT_MATRIX_MAX_PLANECOUNT	= 32; 			///< maximum plane count @ingroup jitter

    // t_matrix_conv_info flags @ingroup jitter
    static const int JIT_MATRIX_CONVERT_CLAMP	= 0x00000001;  ///< not currently used @ingroup jitter
    static const int JIT_MATRIX_CONVERT_INTERP	= 0x00000002;	///< use interpolation @ingroup jitter
    static const int JIT_MATRIX_CONVERT_SRCDIM	= 0x00000004;	///< use source dimensions @ingroup jitter
    static const int JIT_MATRIX_CONVERT_DSTDIM	= 0x00000008;	///< use destination dimensions @ingroup jitter



    static t_symbol* _jit_sym_char = gensym("char");
    static t_symbol* _jit_sym_long = gensym("long");
    static t_symbol* _jit_sym_float32 = gensym("float32");
    static t_symbol* _jit_sym_float64 = gensym("float64");
    static t_symbol* _jit_sym_list = gensym("list");
    static t_symbol* _jit_sym_atom = gensym("atom");

    static t_symbol *_jit_sym_type = gensym("type");
    static t_symbol *_jit_sym_dim = gensym("dim");
    static t_symbol *_jit_sym_planecount = gensym("planecount");
    static t_symbol *_jit_sym_val = gensym("val");
    static t_symbol *_jit_sym_plane = gensym("plane");
    static t_symbol *_jit_sym_cell = gensym("cell");

    static t_symbol* _jit_sym_jit_mop = gensym("jit_mop");
    static t_symbol* _jit_sym_getdata = gensym("getdata");
    static t_symbol* _jit_sym_data = gensym("data");
    static t_symbol* _jit_sym_getindex = gensym("getindex");
    static t_symbol* _jit_sym_getinfo = gensym("getinfo");
    static t_symbol* _jit_sym_setinfo = gensym("setinfo");
    static t_symbol* _jit_sym_setinfo_ex = gensym("setinfo_ex");
    static t_symbol* _jit_sym_lock = gensym("lock");
    static t_symbol* _jit_sym_matrix_calc = gensym("matrix_calc");
    static t_symbol* _jit_sym_jit_matrix = gensym("jit_matrix");
    static t_symbol* _jit_sym_frommatrix = gensym("frommatrix");
    static t_symbol *_jit_sym_outputmatrix	= gensym("outputmatrix");
    static t_symbol *_jit_sym_ioname = gensym("ioname");
    static t_symbol *_jit_sym_matrixname = gensym("matrixname");
    static t_symbol *_jit_sym_outputmode = gensym("outputmode");
    static t_symbol *_jit_sym_matrix = gensym("matrix");
    static t_symbol *_jit_sym_getmatrix = gensym("getmatrix");

    static t_symbol *_jit_sym_inputcount = gensym("inputcount");
    static t_symbol *_jit_sym_outputcount = gensym("outputcount");
    static t_symbol* _jit_sym_getinput = gensym("getinput");
    static t_symbol* _jit_sym_getoutput = gensym("getoutput");
    static t_symbol *_jit_sym_getinputlist	= gensym("getinputlist");
    static t_symbol *_jit_sym_getoutputlist	= gensym("getoutputlist");
    static t_symbol* _jit_sym_mindimcount = gensym("mindimcount");
    static t_symbol* _jit_sym_maxdimcount = gensym("maxdimcount");
    static t_symbol* _jit_sym_minplanecount = gensym("minplanecount");
    static t_symbol* _jit_sym_maxplanecount = gensym("maxplanecount");
    static t_symbol* _jit_sym_dimlink = gensym("dimlink");
    static t_symbol* _jit_sym_planelink = gensym("planelink");
    static t_symbol* _jit_sym_mindim = gensym("mindim");
    static t_symbol* _jit_sym_maxdim = gensym("maxdim");

    static t_symbol* _jit_sym_types = gensym("types");
    static t_symbol* _jit_sym_register = gensym("register");
    static t_symbol *_jit_sym_name = gensym("name");
    static t_symbol *_jit_sym_adapt = gensym("adapt");
    static t_symbol *_jit_sym_decorator = gensym("decorator");
    static t_symbol* _jit_sym_clear = gensym("clear");

    static t_symbol *_jit_sym_nothing		= gensym("");
    static t_symbol *_jit_sym_new			= gensym("new");
    static t_symbol *_jit_sym_free			= gensym("free");
    static t_symbol *_jit_sym_classname		= gensym("classname");
    static t_symbol *_jit_sym_getname		= gensym("getname");
    static t_symbol *_jit_sym_getmethod		= gensym("getmethod");
    static t_symbol *_jit_sym_get 			= gensym("get");
    static t_symbol *_jit_sym_set 			= gensym("set");
    static t_symbol *_jit_sym_setall		= gensym("setall");
    static t_symbol *_jit_sym_rebuilding	= gensym("rebuilding");
    static t_symbol *_jit_sym_modified		= gensym("modified");

    static t_symbol *_jit_sym_class_jit_matrix	= gensym("class_jit_matrix");
    static t_symbol *_jit_sym_class_jit_attribute	= gensym("class_jit_attribute");
    static t_symbol *_jit_sym_jit_attribute		= gensym("jit_attribute");
    static t_symbol *_jit_sym_jit_attr_offset	= gensym("jit_attr_offset");
    static t_symbol *_jit_sym_jit_attr_offset_array	= gensym("jit_attr_offset_array");
    static t_symbol *_jit_sym_max_jit_classex = gensym("max_jit_classex");

    static t_symbol *_jit_sym_position		= gensym("position");
    static t_symbol *_jit_sym_rotatexyz		= gensym("rotatexyz");
    static t_symbol *_jit_sym_scale			= gensym("scale");
    static t_symbol *_jit_sym_quat			= gensym("quat");
    static t_symbol *_jit_sym_direction		= gensym("direction");
    static t_symbol *_jit_sym_lookat		= gensym("lookat");
    static t_symbol *_jit_sym_anim			= gensym("anim");
    static t_symbol *_jit_sym_bounds		= gensym("bounds");
    static t_symbol *_jit_sym_boundcalc		= gensym("boundcalc");
    static t_symbol *_jit_sym_calcbounds	= gensym("calcbounds");
    static t_symbol *_jit_sym_drawto		= gensym("drawto");

    static t_symbol *_jit_sym_jitter		= gensym("jitter");


    struct t_jit_matrix_info {
        long		size = 0;			///< in bytes (0xFFFFFFFF=UNKNOWN)
        t_symbol*	type = nullptr;			///< primitifve type (char, long, float32, or float64)
        long		flags = 0;			///< flags to specify data reference, handle, or tightly packed
        long		dimcount = 0;		///< number of dimensions
        long		dim[JIT_MATRIX_MAX_DIMCOUNT];		///< dimension sizes
        long		dimstride[JIT_MATRIX_MAX_DIMCOUNT]; ///< stride across dimensions in bytes
        long		planecount = 0;		///< number of planes
    };

    typedef t_object 		t_jit_object; 		///< object header @ingroup jitter
    typedef unsigned char	uchar;


    /**
        Provides base pointer and stride for vector operator functions
     */
    struct t_jit_op_info {
        void*	p;			///< base pointer (coerced to appropriate type)
        long 	stride;		///< stride between elements (in type, not bytes)
    };



    BEGIN_USING_C_LINKAGE



    void* jit_class_new(const char* name, method mnew, method mfree, long size, ...);
    t_jit_err jit_class_register(void* c);
    t_jit_err jit_class_addmethod(void* c, method m, const char* name, ...);
    t_jit_err jit_class_addattr(void* c, t_jit_object* attr);
    t_jit_err jit_class_addadornment(void* c, t_jit_object* o);
    t_jit_err jit_class_addinterface(void *c, void *interfaceclass, long byteoffset, long flags);




    void* max_jit_classex_setup(long oboffset);
    void* jit_class_findbyname(t_symbol* classname);
    t_jit_err max_jit_mop_notify(void *x, t_symbol *s, t_symbol *msg);
    t_jit_err max_jit_mop_assist(void* x, void* b, long m, long a, char* s);
    void max_jit_classex_standard_wrap(void* mclass, void* jclass, long flags);

    t_jit_err max_jit_classex_mop_wrap(void* mclass, void* jclass, long flags);		// legacy api
    t_jit_err max_jit_class_mop_wrap(t_class* mclass, t_class* jclass, long flags);	// new api
    t_jit_err max_jit_classex_mop_mproc(void* mclass, void* jclass, void* mproc); //mproc should be method(void* x, void* mop)
    t_jit_err max_jit_mop_setup_simple(void* x, void* o, long argc, t_atom* argv);

    t_jit_err jit_mop_ioproc_copy_adapt(void* mop, void* mop_io, void* matrix);

    t_jit_err jit_mop_single_type(void* x, t_symbol* s);
    t_jit_err jit_mop_single_planecount(void* x, long c);
    t_jit_err jit_mop_input_nolink(void* mop, long c);
    t_jit_err jit_mop_output_nolink(void* mop, long c);

    long max_jit_mop_getoutputmode(void* x);
    void* max_jit_mop_getinput(void* x, long c);
    void* max_jit_mop_getoutput(void* x, long c);
    t_jit_err max_jit_mop_outputmatrix(void *x);
    t_jit_err max_jit_mop_setup(void *x);
    t_jit_err max_jit_mop_inputs(void *x);
    t_jit_err max_jit_mop_inputs_resize(void *x, long count);
    t_jit_err max_jit_mop_outputs(void *x);
    t_jit_err max_jit_mop_outputs_resize(void *x, long count);
    t_jit_err jit_matrix_info_default(t_jit_matrix_info* info);
    t_jit_err max_jit_mop_variable_addinputs(void *x, long c);
    t_jit_err max_jit_mop_variable_addoutputs(void *x, long c);
    t_jit_err max_jit_mop_matrix_args(void *x, long argc, t_atom *argv);

    t_jit_err max_jit_mop_name(void *x, void *attr, long argc, t_atom *argv);
    t_jit_err max_jit_mop_getname(void *x, void *attr, long *argc, t_atom **argv);
    t_jit_err max_jit_mop_type(void *x, void *attr, long argc, t_atom *argv);
    t_jit_err max_jit_mop_gettype(void *x, void *attr, long *argc, t_atom **argv);
    t_jit_err max_jit_mop_dim(void *x, void *attr, long argc, t_atom *argv);
    t_jit_err max_jit_mop_getdim(void *x, void *attr, long *argc, t_atom **argv);
    t_jit_err max_jit_mop_planecount(void *x, void *attr, long argc, t_atom *argv);
    t_jit_err max_jit_mop_getplanecount(void *x, void *attr, long *argc, t_atom **argv);

    void jit_error_code(void* x,t_jit_err v); //interrupt safe


    void jit_parallel_ndim_simplecalc3(method fn, void* data, long dimcount, long* dim, long planecount, t_jit_matrix_info* minfo1, char* bp1,
    t_jit_matrix_info* minfo2, char* bp2, t_jit_matrix_info* minfo3, char* bp3, long flags1, long flags2, long flags3);


    void max_jit_mop_free(void* x);
    void max_jit_object_free(void* x); // new api?

    C74_DEPRECATED( void* max_jit_obex_new(void* mc, t_symbol* classname) );
    void* max_jit_object_alloc(t_class* mc, t_symbol* jitter_classname);

    C74_DEPRECATED( void max_jit_obex_free(void* x) );
    t_jit_err jit_attr_addfilterset_clip(void* x, double min, double max, long usemin, long usemax);
    void max_jit_attr_args(void* x, short ac, t_atom* av);
    long max_jit_attr_args_offset(short ac, t_atom *av);

    t_jit_err max_jit_obex_set(void *x, void *p);
    void* max_jit_obex_jitob_get(void* x);
    void max_jit_obex_jitob_set(void *x, void *jitob);
    void max_jit_obex_dumpout_set(void *x, void *outlet);
    void *max_jit_obex_dumpout_get(void *x);
    void max_jit_obex_dumpout(void *x, const t_symbol *s, short argc, const t_atom *argv);
    void *max_jit_obex_adornment_get(void *x, t_symbol *classname);
    t_jit_err max_jit_obex_addadornment(void *x,void *adornment);
    void max_jit_obex_gimmeback(void *x, const t_symbol *s, long ac, const t_atom *av);
    void max_jit_obex_gimmeback_dumpout(void *x, const t_symbol *s, long ac, const t_atom *av);
    t_jit_err max_jit_obex_proxy_new(void *x, long c);
    long max_jit_obex_inletnumber_get(void *x);
    void max_jit_obex_inletnumber_set(void *x, long inletnumber);

    void* jit_object_alloc(void* c);
    t_jit_object* jit_object_new(t_symbol* classname, ...);
    #ifdef C74_X64
        #define jit_object_new(...) C74_VARFUN(jit_object_new_imp, __VA_ARGS__)
    #endif
    t_jit_object* jit_object_new_imp(void* classname, void* p1, void* p2, void* p3, void* p4, void* p5, void* p6, void* p7, void* p8, void* dummy);
    t_jit_err jit_object_free(void* x);

    // USE THE STANDARD OBJECT_METHOD!
    //void* jit_object_method(void* x, t_symbol* s, ...) JIT_WEAKLINK;
//	void* jit_object_method(void* x, t_symbol* s, ...);
//	#ifdef C74_X64
//		#define jit_object_method(...) C74_VARFUN(c74::max::jit_object_method_imp, __VA_ARGS__)
//	#endif
//	void* jit_object_method_imp(void* x, void* s, void* p1, void* p2, void* p3, void* p4, void* p5, void* p6, void* p7, void* p8);
//	void* jit_object_method_typed(void* x, t_symbol* s, long ac, t_atom* av, t_atom* rv);


    // new t_class API for constructing max wrapper objects using class_new()
    void *max_jit_object_alloc(t_class *mclass, t_symbol *jitter_classname);
    void max_jit_object_free(void *x);
    void max_jit_class_obex_setup(t_class *mclass, long oboffset);
    t_jit_err max_jit_class_addattr(t_class *mclass, void *attr);
    void max_jit_class_wrap_standard(t_class *mclass, t_class *jclass, long flags);
    void max_jit_class_wrap_addmethods(t_class *mclass, t_class *jclass);
    void max_jit_class_wrap_addmethods_flags(t_class *mclass, t_class *jclass, long flags);
    void max_jit_class_wrap_attrlist2methods(t_class *mclass, t_class *jclass);
    void max_jit_class_addmethod_defer(t_class *mclass, method m, const char *s);
    void max_jit_class_addmethod_defer_low(t_class *mclass, method m, const char *s);
    void max_jit_class_addmethod_usurp(t_class *mclass, method m, const char *s);
    void max_jit_class_addmethod_usurp_low(t_class *mclass, method m, const char *s);






    void* jit_object_attach(t_symbol* s, void* x);
    t_jit_err jit_object_detach(t_symbol* s, void* x);

    t_jit_err jit_attr_setlong(void* x, t_symbol* s, t_atom_long c);
    //t_atom_float jit_attr_getfloat(void* x, t_symbol* s);
    t_jit_err jit_attr_setfloat(void* x, t_symbol* s, t_atom_float c);
    //t_symbol* jit_attr_getsym(void* x, t_symbol* s);
    t_jit_err jit_attr_setsym(void* x, t_symbol* s, t_symbol* c);






    void jit_parallel_ndim_simplecalc1(method fn, void *data, long dimcount, long *dim, long planecount, t_jit_matrix_info *minfo1, char *bp1, long flags1);
    void jit_parallel_ndim_simplecalc2(method fn, void *data, long dimcount, long *dim, long planecount, t_jit_matrix_info *minfo1, char *bp1,
        t_jit_matrix_info *minfo2, char *bp2, long flags1, long flags2);
    void jit_parallel_ndim_simplecalc3(method fn, void *data, long dimcount, long *dim, long planecount, t_jit_matrix_info *minfo1, char *bp1,
        t_jit_matrix_info *minfo2, char *bp2, t_jit_matrix_info *minfo3, char *bp3, long flags1, long flags2, long flags3);
    void jit_parallel_ndim_simplecalc4(method fn, void *data, long dimcount, long *dim, long planecount, t_jit_matrix_info *minfo1, char *bp1,
        t_jit_matrix_info *minfo2, char *bp2, t_jit_matrix_info *minfo3, char *bp3, t_jit_matrix_info *minfo4, char *bp4,
        long flags1, long flags2, long flags3, long flags4);













    // jit.gl.chunk.h

    namespace jit_gl_chunk_flags {
        static const unsigned long IGNORE_TEXTURES	= 1 << 0;
        static const unsigned long IGNORE_NORMALS	= 1 << 1;
        static const unsigned long IGNORE_COLORS	= 1 << 2;
        static const unsigned long IGNORE_EDGES		= 1 << 3;
    }

    /// t_jit_glchunk is a public structure to store one gl-command's-worth of data,
    /// in a format which can be passed easily to glDrawRangeElements, and matrixoutput.
    struct t_jit_glchunk {
        t_symbol	*	prim;			///< drawing primitive. "tri_strip", "tri", "quads", "quad_grid", etc.
        t_jit_object *	m_vertex;		///< vertex matrix containing xyzst... data
        t_symbol *		m_vertex_name;	///< vertex matrix name
        t_jit_object *	m_index;		///< optional 1d matrix of vertex indices to use with drawing primitive
        t_symbol *		m_index_name;	///< index matrix name
        unsigned long	m_flags;		///< chunk flags to ignore texture, normal, color, or edge planes when drawing
        void *			next_chunk;		///< pointer to next chunk for drawing a list of chunks together
    };

    t_jit_glchunk * jit_glchunk_new(t_symbol * prim, int planes, int vertices, int indices);
    t_jit_glchunk * jit_glchunk_grid_new(t_symbol * prim, int planes, int width, int height);
    void jit_glchunk_delete(t_jit_glchunk * x);
    t_jit_err jit_glchunk_copy(t_jit_glchunk ** newcopy, t_jit_glchunk * orig);





    // ob3d stuff
    // flags -- default: all flags off.
    namespace jit_ob3d_flags {
        static const unsigned long NO_ROTATION_SCALE		= 1 << 0;      ///< ob3d flag @ingroup jitter
        static const unsigned long NO_POLY_VARS				= 1 << 1;      ///< ob3d flag @ingroup jitter
        static const unsigned long NO_BLEND					= 1 << 2;      ///< ob3d flag @ingroup jitter
        static const unsigned long NO_TEXTURE				= 1 << 3;      ///< ob3d flag @ingroup jitter
        static const unsigned long NO_MATRIXOUTPUT			= 1 << 4;      ///< ob3d flag @ingroup jitter
        static const unsigned long AUTO_ONLY				= 1 << 5;      ///< ob3d flag @ingroup jitter
        static const unsigned long DOES_UI					= 1 << 6;      ///< ob3d flag @ingroup jitter
        static const unsigned long NO_DEPTH					= 1 << 7;      ///< ob3d flag @ingroup jitter
        static const unsigned long NO_ANTIALIAS				= 1 << 8;      ///< ob3d flag @ingroup jitter
        static const unsigned long NO_FOG					= 1 << 9;      ///< ob3d flag @ingroup jitter
        static const unsigned long NO_LIGHTING_MATERIAL		= 1 << 10;     ///< ob3d flag @ingroup jitter
        static const unsigned long HAS_LIGHTS				= 1 << 11;     ///< ob3d flag @ingroup jitter
        static const unsigned long HAS_CAMERA				= 1 << 12;     ///< ob3d flag @ingroup jitter
        static const unsigned long IS_RENDERER				= 1 << 13;     ///< ob3d flag @ingroup jitter
        static const unsigned long NO_COLOR					= 1 << 14;     ///< ob3d flag @ingroup jitter
        static const unsigned long IS_SLAB					= 1 << 15;     ///< ob3d flag @ingroup jitter
        static const unsigned long NO_SHADER				= 1 << 16;
        static const unsigned long PASS_THRU				= 1 << 17;
        static const unsigned long IS_CAMERA				= 1 << 18;
        static const unsigned long NO_BOUNDS				= 1 << 19;
    }


    struct t_jit_class3d {
        long				oboffset; 		// instance byte offset to the ob3d struct.
        long				flags;
        // extensible
    };


    void *jit_ob3d_setup(void * jit_class, long oboffset, long ob3d_flags);
    t_jit_err jit_ob3d_set(void *x, void *p);
    void *jit_ob3d_get(void *x);
    void *jit_ob3d_new(void *x, t_symbol * dest_name);
    void jit_ob3d_free(void *x);
    t_jit_err jit_ob3d_set_context(void *x);
    t_jit_err jit_ob3d_draw_chunk(void *ob3d, t_jit_glchunk * chunk);
    void jit_ob3d_set_viewport(void *v, long x, long y, long width, long height);

//	void max_ob3d_setup(void);					// legacy api
    void max_jit_class_ob3d_wrap(t_class *c);	// newer api

    // attach jit object bearing an ob3d to a max object and its outlet.
    void max_jit_ob3d_attach(void *x, t_jit_object *jit_ob, void *outlet);
    void max_jit_ob3d_detach(void *x);
    t_jit_err max_jit_ob3d_assist(void *x, void *b, long m, long a, char *s);
    t_atom_long max_jit_ob3d_acceptsdrag(void *x, t_object *drag, t_object *view);

    void * ob3d_jitob_get(void *v);
    void * ob3d_patcher_get(void *v);
    void * ob3d_next_get(void *v);
    long ob3d_auto_get(void *v);
    long ob3d_enable_get(void *v);
    long ob3d_ui_get(void *v);
    void * ob3d_outlet_get(void *v);
    long ob3d_dirty_get(void *v);
    void ob3d_dirty_set(void *v, long c);
    void ob3d_dest_dim_set(void *v, long width, long height);
    void ob3d_dest_dim_get(void *v, long *width, long *height);
    void ob3d_render_ptr_set(void *v, void *render_ptr);
    void * ob3d_render_ptr_get(void *v);

    void * ob3d_get_light(void *v, long index);
    void ob3d_set_color(void *v, float *color);
    void ob3d_get_color(void *v, float *color);
    long ob3d_texture_count(void *v);

    t_jit_err ob3d_draw_begin(void *ob3d, long setup);
    t_jit_err ob3d_draw_end(void *ob3d, long setup);
    t_jit_err ob3d_draw_preamble(void *ob3d);

    t_symbol * jit_ob3d_init_jpatcher_render(void *jitob);








    END_USING_C_LINKAGE

}} // namespace c74::max


#include "jit.gl.h"
