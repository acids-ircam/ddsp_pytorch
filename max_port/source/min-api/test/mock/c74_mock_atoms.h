/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include <cassert>

namespace c74 {
namespace max {

    typedef int 				t_int;
    typedef unsigned int 		t_uint; 	///< an unsigned int as defined by the architecture / platform  @ingroup misc
    typedef char 				t_int8; 	///< a 1-byte int  @ingroup misc
    typedef unsigned char 		t_uint8;	///< an unsigned 1-byte int  @ingroup misc
    typedef short 				t_int16; 	///< a 2-byte int  @ingroup misc
    typedef unsigned short 		t_uint16; 	///< an unsigned 2-byte int  @ingroup misc
    typedef int 				t_int32; 	///< a 4-byte int  @ingroup misc
    typedef unsigned int 		t_uint32; 	///< an unsigned 4-byte int  @ingroup misc
    typedef long long 			t_int64;	///< an 8-byte int  @ingroup misc
    typedef unsigned long long 	t_uint64;	///< an unsigned 8-byte int  @ingroup misc
    typedef t_uint32 			t_fourcc; 	///< an integer of suitable size to hold a four char code / identifier  @ingroup misc

#ifdef C74_X64
    typedef unsigned long long	t_ptr_uint;		///< an unsigned pointer-sized int  @ingroup misc
    typedef long long			t_ptr_int; 		///< a pointer-sized int  @ingroup misc
    typedef double				t_atom_float;	///< the type that is an A_FLOAT in a #t_atom  @ingroup misc
#else
    typedef unsigned long		t_ptr_uint;		///< an unsigned pointer-sized int  @ingroup misc
    typedef long 				t_ptr_int; 		///< a pointer-sized int  @ingroup misc
    typedef float 				t_atom_float; 	///< the type that is an A_FLOAT in a #t_atom  @ingroup misc
#endif

    typedef t_ptr_uint 	t_ptr_size;		///< unsigned pointer-sized value for counting (like size_t)  @ingroup misc
    typedef t_ptr_int 	t_atom_long;	///< the type that is an A_LONG in a #t_atom  @ingroup misc
    typedef t_atom_long	t_max_err;		///< an integer value suitable to be returned as an error code  @ingroup misc

    typedef char** t_handle;			///< a handle (address of a pointer)  @ingroup misc
    typedef char* t_ptr;				///< a pointer  @ingroup misc

    typedef t_uint8 t_bool; 			///< a true/false variable  @ingroup misc
    typedef t_int16 t_filepath;			///< i.e. path/vol in file APIs identifying a folder  @ingroup misc


    struct t_object;

    struct t_symbol {
        const char*		s_name;		///< name: a c-string
        t_object*		s_thing;	///< possible binding to a t_object
    };

    MOCK_EXPORT t_symbol* gensym(const char* string);






    // opaque internals used within the t_object
    struct t_messlist;
    struct t_inlet;
    struct t_outlet;
    static const long OB_MAGIC = 1758379419L;

    /// The structure for the head of any first-class object.
    ///	@ingroup obj
    struct t_object {
        t_messlist*	o_messlist;		///<  list of messages and methods. The -1 entry contains a pointer to its #t_class entry.
        t_ptr_int 	o_magic;		///< magic number
        t_inlet*	o_inlet;		///<  list of inlets
        t_outlet*	o_outlet;		///<  list of outlets
    };





    /// officially recognized t_atom types @ingroup atom
    enum e_max_atomtypes : long {
        A_NOTHING = 0,	///< no type, thus no atom
        A_LONG,			///< integer (32-bit on 32-bit arch, 64-bit on 64-bit arch)
        A_FLOAT,		///< decimal (float on 32-bit arch, double on 64-bit arch)
        A_SYM,			///< symbol
        A_OBJ,			///< object
        A_DEFLONG,		///< long but defaults to zero
        A_DEFFLOAT,		///< float, but defaults to zero
        A_DEFSYM,		///< symbol, defaults to ""
        A_GIMME,		///< request that args be passed as an array, the routine will check the types itself.
        A_CANT,			///< cannot typecheck args
        A_SEMI,			///< semicolon
        A_COMMA,		///< comma
        A_DOLLAR,		///< dollar
        A_DOLLSYM,		///< dollar
        A_GIMMEBACK,	///< request that args be passed as an array, the routine will check the types itself. can return atom value in final atom ptr arg. function returns long error code 0 = no err. see gimmeback_meth typedef

        A_DEFER = 0x41,	///< A special signature for declaring methods. This is like A_GIMME, but the call is deferred.
        A_USURP = 0x42,	///< A special signature for declaring methods. This is like A_GIMME, but the call is deferred and multiple calls within one servicing of the queue are filtered down to one call.
        A_DEFER_LOW = 0x43,	///< A special signature for declaring methods. This is like A_GIMME, but the call is deferref to the back of the queue.
        A_USURP_LOW = 0x44	///< A special signature for declaring methods. This is like A_GIMME, but the call is deferred to the back of the queue and multiple calls within one servicing of the queue are filtered down to one call.

    };

    /// Union for packing any of the datum defined in #e_max_atomtypes. @ingroup atom
    union word {
        t_atom_long		w_long;		///< long integer
        t_atom_float	w_float;	///< 32-bit float
        t_symbol*		w_sym;		///< pointer to a symbol in the Max symbol table
        t_object*		w_obj;		///< pointer to a #t_object or other generic pointer
    };

    /// An atom is a typed datum. @ingroup atom
    struct t_atom {
        short			a_type;
        union word		a_w;
    };



    // Mock implementations for basic atom accessors

    MOCK_EXPORT long atom_gettype(const t_atom *a)			{ return a->a_type; }

    MOCK_EXPORT t_atom_float atom_getfloat(const t_atom *a)	{
        if (a->a_type == A_FLOAT)
            return a->a_w.w_float;
        else if (a->a_type == A_LONG)
            return static_cast<t_atom_float>(a->a_w.w_long);
        else {
            assert(false); // not a number
            return 0;
        }
    }
    MOCK_EXPORT t_atom_long atom_getlong(const t_atom *a)	{ return a->a_w.w_long; }
    MOCK_EXPORT t_symbol* atom_getsym(const t_atom *a)		{ return a->a_w.w_sym; }
    MOCK_EXPORT t_object* atom_getobj(const t_atom *a)		{ return a->a_w.w_obj; }

    MOCK_EXPORT t_max_err atom_setfloat(t_atom *a, double v)		{a->a_w.w_float = v; a->a_type = A_FLOAT; return 0;}
    MOCK_EXPORT t_max_err atom_setlong(t_atom *a, t_atom_long v)	{a->a_w.w_long = v; a->a_type = A_LONG; return 0;}
    MOCK_EXPORT t_max_err atom_setsym(t_atom *a, t_symbol *s)		{a->a_w.w_sym = s; a->a_type = A_SYM; return 0;}
    MOCK_EXPORT t_max_err atom_setobj(t_atom *a, t_object *o)		{ a->a_w.w_obj = o; a->a_type = A_OBJ; return 0; }

    static const int ATOM_GETTEXT_DEFAULT_SIZE = 64;		// a reasonable start?
    static const int ATOM_GETTEXT_MAX_NUM_SIZE = 1024;

    MOCK_EXPORT t_max_err atom_gettext(long ac, t_atom *av, long *textsize, const char **text, long flags)
    {
        // TODO: return something reasonable?  Strip out all usage?  Use code from kernel?
        *text = "";
        *textsize = 0;
        return 0;
    }


// Atom Arrays

    struct t_atomarray;

    MOCK_EXPORT t_atomarray* atomarray_new(long ac, t_atom* av) { return nullptr; }
    MOCK_EXPORT void atomarray_appendatom(t_atomarray* x, t_atom* a) {};



// Special stuff for the mocked testing environment

/**	A vector of atoms.	*/
typedef std::vector<t_atom>	t_atom_vector;

/** A sequence of atom vectors.
    Used to log inlet and outlet activity in the mock environment.
    We can use logging of inlet and outlet sequences for BDD.
    We can also do more traditional state-based testing.
    Or mix-n-match as we see fit...

    @remark		should sequences have time-stamps?
 */
typedef std::vector<t_atom_vector>	t_sequence;


/** Expose t_atom for use in std output streams.
    @remark		Would be nice to have the functionality of atoms_totext(), but that's also pretty complex!
 */
template <class charT, class traits>
std::basic_ostream <charT, traits>& operator<< (std::basic_ostream <charT, traits>& stream, const t_atom& a)
{
    char str[4096]; // TODO: can I really alloc this on the stack? the stream probably copies it, but not sure!

    if (a.a_type == A_LONG)
#ifdef _LP64
        snprintf(str, 4096, "%lld", a.a_w.w_long);
#else
        snprintf(str, 4096, "%ld", a.a_w.w_long);
#endif
    else if (a.a_type == A_FLOAT)
        snprintf(str, 4096, "%f", a.a_w.w_float);
    else if (a.a_type == A_SYM)
        snprintf(str, 4096, "%s", a.a_w.w_sym->s_name);
    else
        snprintf(str, 4096, "%s", "<nonsense>");

    return stream << str;
}


}} // namespace c74::max
