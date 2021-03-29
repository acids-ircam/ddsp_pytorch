/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {


    /** The atomarray object. @ingroup atomarray */
    typedef t_object t_atomarray;


    BEGIN_USING_C_LINKAGE

    /**
        Create a new atomarray object.
        Note that atoms provided to this function will be <em>copied</em>. The copies stored internally to the atomarray instance.
        You can free the atomarray by calling object_free().

        @ingroup	atomarray
        @param	ac	The number of atoms to be initially contained in the atomarray.
        @param	av	A pointer to the first of an array of atoms to initially copy into the atomarray.
        @return		Pointer to the new atomarray object.

        @remark		Note that due to the unusual prototype of this method that you cannot instantiate this object using the
                    object_new_typed() function.  If you wish to use the dynamically bound creator to instantiate the object,
                    you should instead should use object_new() as demonstrated below.  The primary reason that you might choose
                    to instantiate an atomarray using object_new() instead of atomarray_new() is for using the atomarray object
                    in code that is also intended to run in Max 4.
        @code
        object_new(CLASS_NOBOX, gensym("atomarray"), argc, argv);
        @endcode

        @see		atomarray_duplicate()
    */
    t_atomarray* atomarray_new(long ac, t_atom* av);

    /**
        Set the atomarray flags.

        @ingroup	atomarray

        @param	x		The atomarray instance.
        @param  flags	The new value for the flags.
    */
    void atomarray_flags(t_atomarray* x, long flags);

    /**
        Get the atomarray flags.

        @ingroup	atomarray

        @param	x	The atomarray instance.
        @return		The current value of the atomarray flags.
    */
    long atomarray_getflags(t_atomarray* x);

    /**
        Replace the existing array contents with a new set of atoms
        Note that atoms provided to this function will be <em>copied</em>.  The copies stored internally to the atomarray instance.

        @ingroup	atomarray

        @param	x	The atomarray instance.
        @param	ac	The number of atoms to be initially contained in the atomarray.
        @param	av	A pointer to the first of an array of atoms to initially copy into the atomarray.
        @return		A Max error code.
    */
    t_max_err atomarray_setatoms(t_atomarray* x, long ac, t_atom* av);


    /**
        Retrieve a pointer to the first atom in the internal array of atoms.
        This method does not copy the atoms, btu simply provides access to them.
        To retrieve a copy of the atoms use atomarray_copyatoms().

        @ingroup	atomarray

        @param	x	The atomarray instance.
        @param	ac	The address of a long where the number of atoms will be set.
        @param	av	The address of a #t_atom pointer where the address of the first atom of the array will be set.
        @return		A Max error code.

        @see		atomarray_copyatoms()
    */
    t_max_err atomarray_getatoms(t_atomarray* x, long* ac, t_atom** av);


    /**
        Retrieve a copy of the atoms in the array.
        To retrieve a pointer to the contained atoms use atomarray_getatoms().

        @ingroup	atomarray

        @param	x	The atomarray instance.
        @param	ac	The address of a long where the number of atoms will be set.
        @param	av	The address of a #t_atom pointer where the atoms will be allocated and copied.
        @return		A Max error code.

        @remark		You are responsible for freeing memory allocated for the copy of the atoms returned.
        @code
        long	ac = 0;
        t_atom* av = NULL;

        atomarray_copyatoms(anAtomarray, &ac, &av);
        if(ac && av){
            // do something with ac and av here...
            sysmem_freeptr(av);
        }
        @endcode

        @see		atomarray_getatoms()
    */
    t_max_err atomarray_copyatoms(t_atomarray* x, long* ac, t_atom** av);


    /**
        Return the number of atoms in the array.

        @ingroup	atomarray
        @param	x	The atomarray instance.
        @return		The number of atoms in the array.
    */
    t_atom_long atomarray_getsize(t_atomarray* x);


    /**
        Copy an a specific atom from the array.

        @ingroup		atomarray
        @param	x		The atomarray instance.
        @param	index	The zero-based index into the array from which to retrieve an atom pointer.
        @param	av		The address of an atom to contain the copy.
        @return			A Max error code.

        @remark			Example:
        @code
        {
            t_atom a;

            // fetch a copy of the second atom in a previously existing array
            atomarray_getindex(anAtomarray, 1, &a);
            // do something with the atom here...
        }
        @endcode
    */
    t_max_err atomarray_getindex(t_atomarray* x, long index, t_atom* av);


    // not exported yet
    t_max_err atomarray_setindex(t_atomarray* x, long index, t_atom* av);


    /**
        Create a new atomarray object which is a copy of another atomarray object.

        @ingroup		atomarray
        @param	x		The atomarray instance which is to be copied.
        @return			A new atomarray which is copied from x.

        @see	atomarray_new()
    */
    void* atomarray_duplicate(t_atomarray* x);


    /**
        Copy a new atom onto the end of the array.

        @ingroup		atomarray
        @param	x		The atomarray instance.
        @param	a		A pointer to the new atom to append to the end of the array.

        @see	atomarray_appendatoms()
        @see	atomarray_setatoms()
    */
    void atomarray_appendatom(t_atomarray* x, t_atom* a);


    /**
        Copy multiple new atoms onto the end of the array.

        @ingroup		atomarray
        @param	x		The atomarray instance.
        @param	ac		The number of new atoms to be appended to the array.
        @param	av		A pointer to the first of the new atoms to append to the end of the array.

        @see	atomarray_appendatom()
        @see	atomarray_setatoms()
    */
    void atomarray_appendatoms(t_atomarray* x, long ac, t_atom* av);


    /**
        Remove an atom from any location within the array.
        The array will be resized and collapsed to fill in the gap.

        @ingroup		atomarray
        @param	x		The atomarray instance.
        @param	index	The zero-based index of the atom to remove from the array.
    */
    void atomarray_chuckindex(t_atomarray* x, long index);


    /**
        Clear the array.  Frees all of the atoms and sets the size to zero.
        This function does not perform a 'deep' free, meaning that any #A_OBJ atoms will not have their object's freed.
        Only the references to those objects contained in the atomarray will be freed.

        @ingroup	atomarray
        @param	x	The atomarray instance.
    */
    void atomarray_clear(t_atomarray* x);


    /**
        Call the specified function for every item in the atom array.

        @ingroup atomarray
        @param	x		The atomarray instance.
        @param	fun		The function to call, specified as function pointer cast to a Max #method.
        @param	arg		An argument that you would like to pass to the function being called.

        @remark			The atomarray_funall() method will call your function for every item in the list.
                        It will pass both a pointer to the item in the list, and any argument that you
                        provide.  The following example shows a function that could be called by hashtab_funall().
        @code
        void myFun(t_atom* a, void* myArg)
        {
            // do something with a and myArg here
            // a is the atom in the atom array
        }
        @endcode

        @see			linklist_funall()
        @see			hashtab_funall()
    */
    void atomarray_funall(t_atomarray* x, method fun, void* arg);

    END_USING_C_LINKAGE

}} // namespace c74::max
