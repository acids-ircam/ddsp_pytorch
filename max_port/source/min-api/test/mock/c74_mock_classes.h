/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {

    typedef void* (*method)(void*, ...);	///< Function pointer type for generic methods. @ingroup datatypes

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
        CLASS_FLAG_BOX = 0x00000001L,	///< for use in a patcher
        CLASS_FLAG_POLYGLOT = 0x00000002L,	///< for use by any text language (c/js/java/etc)
        CLASS_FLAG_NEWDICTIONARY = 0x00000004L,	///< dictionary based constructor
        CLASS_FLAG_REGISTERED = 0x00000008L,	///< for backward compatible messlist implementation (once reg'd can't grow)
        CLASS_FLAG_UIOBJECT = 0x00000010L,	///< for objects that don't go inside a newobj box.
        CLASS_FLAG_ALIAS = 0x00000020L,	///< for classes that are just copies of some other class (i.e. del is a copy of delay)
        CLASS_FLAG_DO_NOT_PARSE_ATTR_ARGS = 0x00000080L, 	///< override dictionary based constructor attr arg parsing
        CLASS_FLAG_DO_NOT_ZERO = 0x00000100L, 	///< don't zero the object struct on construction (for efficiency)
        CLASS_FLAG_NOATTRIBUTES = 0x00010000L,	///< for efficiency
        CLASS_FLAG_OWNATTRIBUTES = 0x00020000L,	///< for classes which support a custom attr interface (e.g. jitter)
        CLASS_FLAG_PARAMETER = 0x00040000L,	///< for classes which have a parameter
        CLASS_FLAG_RETYPEABLE = 0x00080000L,	///< object box can be retyped without recreating the object
        CLASS_FLAG_OBJECT_METHOD = 0x00100000L		///< objects of this class may have object specific methods
    };

/**	A mocked messlist implementation.
    Unlike the real Max messlist this uses an STL hashtab container to manage all of memory and accessors.
 */
typedef std::unordered_map<std::string, method>	t_mock_messlist;




MOCK_EXPORT t_symbol *object_classname(t_object *x);

/**	Initializes a class by informing Max of its name, instance creation and free functions, size and argument types.
    This mocks the behavior of Max's real class_addmethod().

    @ingroup classes
    @param 	name	The class's name, as a C-string
    @param 	mnew	The instance creation function
    @param 	mfree	The instance free function
    @param 	size	The size of the object's data structure in bytes.
    @param 	mmenu	Obsolete - pass NULL.
    @param 	type	A NULL-terminated list of types for the constructor.
    @return 		This function returns the class pointer for the new object class.

    @remark			The internal messlist implementation for the mock t_class is slightly different than that of the real Max.
                    Generally this should not trip anyone up, but in the specific case where the classname is expect to be in the
                    messlist at index -1 there will be problems as this messlist is a hashtab and not an array.
                    The code for this function and object_classname() should make be self-evident for how to accomplish this using the mock t_class.
 */
MOCK_EXPORT t_class* class_new(const char* name, const method mnew, const method mfree, long size, const method mmenu, short type, ...) {
    t_class			*c = new t_class;
    t_mock_messlist	*mock_messlist = new t_mock_messlist;

    c->c_sym = gensym(name);
    c->c_freefun = mfree;
    c->c_size = size;

    (*mock_messlist)["###CLASS###"] = (method)c;
    (*mock_messlist)["classname"] = (method)object_classname;
    c->c_messlist = (t_messlist*)mock_messlist;

    return c;
}


MOCK_EXPORT void* jit_class_new(const char* name, method mnew, method mfree, long size, ...)
{
    // TODO: is this the right thing to do?
    return class_new(name, mnew, mfree, size, nullptr, 0);
}



/**	Add a class to the class registry.
    This should mock the behavior of Max's real class_register() but currently does nothing at all!

    @ingroup	classes
    @param		name_space	Typically "box".
    @param		c			The class to register.
    @return					An error code.

    @remark		For an implementation that works with object_new() et. al. we need to actually implement something here.
                The way it works in Max itself is that we have a registry at global scope.
                Ideally we could implement this non-globally, as suggested also in the code surrounding the mock gensym() implementation.
 */
MOCK_EXPORT t_max_err class_register(t_symbol *name_space, t_class *c) { return 0; }


/**	Add a method to a #t_class.
    This mocks the behavior of Max's real class_addmethod().

    @ingroup	classes
    @param		c		The class to which to add the message binding.
    @param		m		The method to which the message maps.
    @param		name	The message name.
    @param		...		A list of types -- NOT CURRENTLY IMPLEMENTED
    @return				A Max error code.

    @seealso	class_new()
 */
MOCK_EXPORT t_max_err class_addmethod(t_class *c, const method m, const char *name, ...)
{
    t_mock_messlist *mock_messlist = (t_mock_messlist*)c->c_messlist;
    (*mock_messlist)[name] = m;
    return 0;
}

MOCK_EXPORT t_max_err class_addattr(t_class *x, t_object* attr)
{
    // TODO: implement
    return 0;
}


}} // namespace c74::max

#include "c74_mock_inlets.h"
#include "c74_mock_outlets.h"

namespace c74 {
namespace max {


/**	Create an instance of a #t_class.
    This mocks the behavior of Max's real object_alloc().

    @ingroup	classes
    @param	c	The class of which to create an instance.
    @return		A pointer to the instance.

    @remark		At the moment this implementation does not know about fancy obex stuff!
 */
    MOCK_EXPORT void *object_alloc(t_class *c)
{
    t_object *o = (t_object*)malloc(c->c_size);

    o->o_messlist = c->c_messlist;
    o->o_magic = OB_MAGIC;
    o->o_inlet = (t_inlet*) new object_inlets(o);
    {
        t_mock_outlets	mock_outlets;

        // outlets are accessed through a global hash rather than by this struct member
        // this is because many outlet calls do not include the t_object pointer!
        // o->o_outlet = (struct outlet*) new t_mock_outlets;
        g_object_to_outletset[o] = mock_outlets;
    }
    return o;
}


/**	Free an instance of a #t_class.
    This mocks the behavior of Max's real object_alloc().

     @ingroup	classes
     @param	x	The pointer to the object to free.
     @return	An error code.

     @remark	At the moment, we don't know about tinyobjects, should be easy to add support for that.
 */
MOCK_EXPORT t_max_err object_free(void *x) {
    auto o = (t_object*)x;
    if (o && o->o_magic == OB_MAGIC) {
        t_mock_messlist *mock_messlist = (t_mock_messlist*)o->o_messlist;
        t_class			*c = (t_class *) ((*mock_messlist)["###CLASS###"]);

        if (c->c_freefun)
            (*c->c_freefun)(x);
        o->o_magic = -1;

        delete (object_inlets*)o->o_inlet;
        g_object_to_outletset.erase(o);

        free(x);
    }
    return 0;
}


/**	Return the name of the class represented by a #t_object instance.
    This mocks the behavior of Max's real object_classname().
 */
MOCK_EXPORT t_symbol *object_classname(t_object *x)
{
    t_object		*o = (t_object*)x;
    t_mock_messlist *mock_messlist = (t_mock_messlist*)o->o_messlist;
    t_class			*c = (t_class *) ((*mock_messlist)["###CLASS###"]);

    return c->c_sym;
}


MOCK_EXPORT long object_classname_compare(void* x, t_symbol* name) {
    return false;
}



MOCK_EXPORT method zgetfn(t_object *op, t_symbol *msg)
{
    t_mock_messlist *messlist = (t_mock_messlist*)op->o_messlist;

    return (*messlist)[msg->s_name];
}


inline void* object_method(t_object* target_object, t_symbol* method_name, void* arg1, void* arg2, void* arg3, void* arg4, void* arg5) {
    method m = zgetfn(target_object, method_name);
    if (m)
        return m(target_object, arg1, arg2, arg3, arg4, arg5);
    else
        return nullptr;
}


MOCK_EXPORT method object_method_direct_getmethod(t_object *x, t_symbol *sym)
{
    // TODO: This function should be an obex-enhanced version of zgetfn(), but the mock implementation is currently obex crippled.
    return zgetfn(x, sym);
}


MOCK_EXPORT t_object *object_method_direct_getobject(t_object *x, t_symbol *sym)
{
    // TODO: once again, the mock implementation is not currently obex-savvy
    return x;
}


}} // namespace c74::max
