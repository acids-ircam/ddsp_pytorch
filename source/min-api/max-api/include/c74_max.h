/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#define C74_MAX_SDK_VERSION 0x0730

#include "c74_max_platform.h"
#include "c74_max_preprocessor.h"
#include "c74_max_sysmem.h"


#if C74_PRAGMA_STRUCT_PACKPUSH
    #pragma pack(push, 2)
#endif


#include "c74_max_symbol.h"

namespace c74 {
namespace max {

    typedef void* (*method)(void* , ...);    ///< Function pointer type for generic methods. @ingroup datatypes

    // opaque internals used within the t_object
    typedef struct messlist {
        t_symbol *m_sym;        ///< Name of the message
        method m_fun;                ///< Method associated with the message
        char m_type[8];    ///< Argument type information
    } t_messlist;

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

    ///	Returns true if a pointer is not a valid object.
    /// @ingroup obj
    inline bool OB_INVALID(t_object* x) {
        #ifdef WIN_VERSION
                return IsBadReadPtr((void*)x,sizeof(t_object)) || x->o_magic != OB_MAGIC;
        #else
                return !x || x->o_magic != OB_MAGIC;
        #endif
    }

    BEGIN_USING_C_LINKAGE

    ///	Retrieves an object instance's class name
    /// @ingroup obj
    ///	@param 	x		The object instance whose class name is being queried
    ///	@return 		The classname, or NULL if unsuccessful.
    t_symbol* object_classname(t_object* x);

    /// Return the namespace to which this object's class belongs
    /// @ingroup obj
    t_symbol* object_namespace(t_object* x);

    END_USING_C_LINKAGE

}} // namespace c74::max

#include "c74_max_atom.h"

namespace c74 {
namespace max {

    /// The data structure for a Max class.
    /// Should be considered opaque, but is not for legacy reasons.
    ///	@ingroup class
    struct t_class {
        t_symbol*	c_sym;				// symbol giving name of class
        t_object**	c_freelist;			// linked list of free ones
        method 		c_freefun;			// function to call when freeing
        t_ptr_uint 	c_size;				// size of an instance
        char 		c_tiny;				// flag indicating tiny header
        char 		c_noinlet;			// flag indicating no first inlet for patcher
        t_symbol*	c_filename;			// name of file associated with this class
        t_messlist*	c_newmess;			// constructor method/type information
        method 		c_menufun;			// function to call when creating from object pallette (default constructor)
        long 		c_flags;			// class flags used to determine context in which class may be used
        long 		c_messcount;		// size of messlist array
        t_messlist*	c_messlist;			// messlist arrray
        void*		c_attributes;		// t_hashtab object
        void*		c_extra;			// t_hashtab object
        long 		c_obexoffset;		// if non zero, object struct contains obex pointer at specified byte offset
        void*		c_methods;			// methods t_hashtab object
        method 		c_attr_get;			// if not set, NULL, if not present CLASS_NO_METHOD
        method 		c_attr_gettarget;	// if not set, NULL, if not present CLASS_NO_METHOD
        method 		c_attr_getnames;	// if not set, NULL, if not present CLASS_NO_METHOD
        t_class*	c_superclass;		// a superclass pointer if this is a derived class
    };


    /** Class flags. If not box or polyglot, class is only accessible in C via known interface
        @ingroup class
    */
    enum e_max_class_flags {
        CLASS_FLAG_BOX =					0x00000001L,	///< for use in a patcher
        CLASS_FLAG_POLYGLOT =				0x00000002L,	///< for use by any text language (c/js/java/etc)
        CLASS_FLAG_NEWDICTIONARY =			0x00000004L,	///< dictionary based constructor
        CLASS_FLAG_REGISTERED =				0x00000008L,	///< for backward compatible messlist implementation (once reg'd can't grow)
        CLASS_FLAG_UIOBJECT =				0x00000010L,	///< for objects that don't go inside a newobj box.
        CLASS_FLAG_ALIAS =					0x00000020L,	///< for classes that are just copies of some other class (i.e. del is a copy of delay)
        CLASS_FLAG_DO_NOT_PARSE_ATTR_ARGS =	0x00000080L, 	///< override dictionary based constructor attr arg parsing
        CLASS_FLAG_DO_NOT_ZERO =			0x00000100L, 	///< don't zero the object struct on construction (for efficiency)
        CLASS_FLAG_NOATTRIBUTES =			0x00010000L,	///< for efficiency
        CLASS_FLAG_OWNATTRIBUTES =			0x00020000L,	///< for classes which support a custom attr interface (e.g. jitter)
        CLASS_FLAG_PARAMETER =				0x00040000L,	///< for classes which have a parameter
        CLASS_FLAG_RETYPEABLE =				0x00080000L,	///< object box can be retyped without recreating the object
        CLASS_FLAG_OBJECT_METHOD =			0x00100000L		///< objects of this class may have object specific methods
    };


    typedef t_object t_patcher;		///< A patcher @ingroup patcher
    typedef t_object t_box;			///< A box @ingroup patcher
    typedef t_object t_clock;		///< A clock @ingroup clocks
    typedef void* t_qelem;			///< A qelem @ingroup qelems



    /// patcher iteration flags @ingroup patcher
    enum {
        PI_DEEP = 1,				///< descend into subpatchers (not used by audio library)
        PI_REQUIREFIRSTIN = 2,		///< if b->b_firstin is NULL, do not call function
        PI_WANTBOX = 4,				///< instead, of b->b_firstin, pass b to function, whether or not b->b_firstin is NULL
        PI_SKIPGEN = 8,
        PI_WANTPATCHER = 16
    };


    BEGIN_USING_C_LINKAGE

    /// Check to see if the function currently being executed is in the main thread.
    ///	@ingroup	threading
    ///	@return		Returns true if the function is being executed in the main thread, otherwise false.
    short systhread_ismainthread(void);

    /// Check to see if the function currently being executed is in a scheduler thread.
    ///	@ingroup	threading
    ///	@return		Returns true if the function is being executed in a scheduler thread, otherwise false.
    short systhread_istimerthread(void);

    /// Check to see if the function currently being executed is in an audio thread.
    /// @ingroup	threading
    /// @return		Returns true if the function is being executed in an audio thread, otherwise false.
    short systhread_isaudiothread(void);

    END_USING_C_LINKAGE


    static const int ASSIST_STRING_MAXSIZE = 256;

    // 1700 is VS2012
    #if __cplusplus <= 199711L && defined(WIN_VERSION) && _MSC_VER < 1700
        typedef long t_assist_function;
    #else
        enum t_assist_function : long {
            ASSIST_INLET = 1,
            ASSIST_OUTLET
        };
    #endif


#ifndef C74_MIN_API
    template <typename T>
    T clamp(T input, T low, T high) {
        return std::min(std::max(input, low), high);
    }
#endif


}} // namespace c74::max




#include "c74_max_systime.h"
#include "c74_max_path.h"
#include "c74_max_proto.h"
#include "c74_max_dictionary.h"
#include "c74_max_object.h"
#include "c74_max_dictobj.h"
#include "c74_max_time.h"



#if C74_PRAGMA_STRUCT_PACKPUSH
    #pragma pack(pop)
#endif



BEGIN_USING_C_LINKAGE

///	ext_main() is the global entry point for an extern to be loaded, which all externs must implement
///	this shared/common prototype ensures that it will be exported correctly on all platforms.
///	@ingroup	class
///	@param		r	Pointer to resources for the external, if applicable.
///	@see  		Anatomy of an Object
///	@version 	Introduced in Max 6.1.9
void C74_EXPORT ext_main(void* r);

END_USING_C_LINKAGE



