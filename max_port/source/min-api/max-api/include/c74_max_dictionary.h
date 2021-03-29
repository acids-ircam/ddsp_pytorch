/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_max_obstring.h"
#include "c74_max_atomarray.h"
#include "c74_max_hashtab.h"

namespace c74 {
namespace max {

    BEGIN_USING_C_LINKAGE


    /** A dictionary entry.
        @ingroup dictionary
        @see t_dictionary
    */
        struct t_dictionary_entry;

    /** The dictionary object.
        @ingroup dictionary
        @see t_dictionary
    */
    typedef t_object t_dictionary;


    /**	Create a new dictionary object.
        You can free the dictionary by calling object_free().
        However, you should keep in mind the guidelines provided in @ref when_to_free_a_dictionary.

        @ingroup dictionary
        @return  Pointer to the new dictionary object.

        @see				object_free()
    */
    t_dictionary* dictionary_new(void);


    /**
        Add a long integer value to the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The name of the key used to index the new value.
                        All keys must be unique.  If the key name already exists,
                        then the existing value associated with the key will be freed prior to the new value's assignment.
        @param	value	The new value to append to the dictionary.
        @return			A Max error code.
    */
    t_max_err dictionary_appendlong(t_dictionary* d, t_symbol* key, t_atom_long value);


    /**
        Add a double-precision float value to the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The name of the key used to index the new value.
                        All keys must be unique.  If the key name already exists,
                        then the existing value associated with the key will be freed prior to the new value's assignment.
        @param	value	The new value to append to the dictionary.
        @return			A Max error code.
    */
    t_max_err dictionary_appendfloat(t_dictionary* d, t_symbol* key, double value);


    /**
        Add a #t_symbol* value to the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The name of the key used to index the new value.
                        All keys must be unique.  If the key name already exists,
                        then the existing value associated with the key will be freed prior to the new value's assignment.
        @param	value	The new value to append to the dictionary.
        @return			A Max error code.
    */
    t_max_err dictionary_appendsym(t_dictionary* d, t_symbol* key, t_symbol* value);


    /**
        Add a #t_atom* value to the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The name of the key used to index the new value.
                        All keys must be unique.  If the key name already exists,
                        then the existing value associated with the key will be freed prior to the new value's assignment.
        @param	value	The new value to append to the dictionary.
        @return			A Max error code.
    */
    t_max_err dictionary_appendatom(t_dictionary* d, t_symbol* key, t_atom* value);


    /**
        Add a C-string to the dictionary.  Internally this uses the #t_symbol object.
        It is useful to use the #t_string in dictionaries rather than the #t_symbol
        to avoid bloating Max's symbol table unnecessarily.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The name of the key used to index the new value.
                        All keys must be unique.  If the key name already exists,
                        then the existing value associated with the key will be freed prior to the new value's assignment.
        @param	value	The new value to append to the dictionary.
        @return			A Max error code.
    */
    t_max_err dictionary_appendstring(t_dictionary* d, t_symbol* key, const char* value);


    /**
        Add an array of atoms to the dictionary.
        Internally these atoms will be copied into a #t_atomarray object, which will be appended to the dictionary
        with the given key.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The name of the key used to index the new value.
                        All keys must be unique.  If the key name already exists,
                        then the existing value associated with the key will be freed prior to the new value's assignment.
        @param	argc	The number of atoms to append to the dictionary.
        @param	argv	The address of the first atom in the array to append to the dictionary.
        @return			A Max error code.
    */
    t_max_err dictionary_appendatoms(t_dictionary* d, t_symbol* key, long argc, t_atom* argv);


    /**
        Add an @ref atomarray object to the dictionary.
        Note that from this point on that you should not free the #t_atomarray*, because the atomarray is now owned by
        the dictionary, and freeing the dictionary will free the atomarray as discussed in @ref when_to_free_a_dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The name of the key used to index the new value.
                        All keys must be unique.  If the key name already exists,
                        then the existing value associated with the key will be freed prior to the new value's assignment.
        @param	value	The new value to append to the dictionary.
        @return			A Max error code.
    */
    t_max_err dictionary_appendatomarray(t_dictionary* d, t_symbol* key, t_object* value);


    /**
        Add a dictionary object to the dictionary.
        Note that from this point on that you should not free the #t_dictionary* that is being added,
        because the newly-added dictionary is now owned by the dictionary to which it has been added,
        as discussed in @ref when_to_free_a_dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The name of the key used to index the new value.
                        All keys must be unique.  If the key name already exists,
                        then the existing value associated with the key will be freed prior to the new value's assignment.
        @param	value	The new value to append to the dictionary.
        @return			A Max error code.
    */
    t_max_err dictionary_appenddictionary(t_dictionary* d, t_symbol* key, t_object* value);


    /**
        Add an object to the dictionary.
        Note that from this point on that you should not free the #t_object* that is being added,
        because the newly-added object is now owned by the dictionary to which it has been added,
        as discussed in @ref when_to_free_a_dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The name of the key used to index the new value.
                        All keys must be unique.  If the key name already exists,
                        then the existing value associated with the key will be freed prior to the new value's assignment.
        @param	value	The new value to append to the dictionary.
        @return			A Max error code.
    */
    t_max_err dictionary_appendobject(t_dictionary* d, t_symbol* key, t_object* value);


    /**
        Retrieve a long integer from the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @return			A Max error code.
    */
    t_max_err dictionary_getlong(const t_dictionary* d, t_symbol* key, t_atom_long* value);


    /**
        Retrieve a double-precision float from the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @return			A Max error code.
    */
    t_max_err dictionary_getfloat(const t_dictionary* d, t_symbol* key, double* value);


    /**
        Retrieve a #t_symbol* from the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @return			A Max error code.
    */
    t_max_err dictionary_getsym(const t_dictionary* d, t_symbol* key, t_symbol** value);


    /**
        Copy a #t_atom from the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @return			A Max error code.
    */
    t_max_err dictionary_getatom(const t_dictionary* d, t_symbol* key, t_atom* value);


    /**
        Retrieve a C-string pointer from the dictionary.
        The retrieved pointer references the string in the dictionary, it is not a copy.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @return			A Max error code.
    */
    t_max_err dictionary_getstring(const t_dictionary* d, t_symbol* key, const char** value);


    /**
        Retrieve the address of a #t_atom array of in the dictionary.
        The retrieved pointer references the t_atoms in the dictionary.
        To fetch a copy of the t_atoms from the dictionary, use dictionary_copyatoms().

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	argc	The address of a variable to hold the number of atoms in the array.
        @param	argv	The address of a variable to hold a pointer to the first atom in the array.
        @return			A Max error code.

        @see			dictionary_copyatoms()
        @see			dictionary_getatoms_ext()
    */
    t_max_err dictionary_getatoms(const t_dictionary* d, t_symbol* key, long* argc, t_atom** argv);

    /**
        Retrieve the address of a #t_atom array in the dictionary.
        The retrieved pointer references the t_atoms in the dictionary.
        Optionally convert strings to symbols.
        To fetch a copy of the t_atoms from the dictionary, use dictionary_copyatoms().

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	stringstosymbols		The flag to convert strings to symbols (true,false).
        @param	argc	The address of a variable to hold the number of atoms in the array.
        @param	argv	The address of a variable to hold a pointer to the first atom in the array.
        @return			A Max error code.

        @see			dictionary_copyatoms()
        @see			dictionary_getatoms()
    */
    t_max_err dictionary_getatoms_ext(const t_dictionary* d, t_symbol* key, long stringstosymbols, long* argc, t_atom** argv);

    /**
        Retrieve copies of a #t_atom array in the dictionary.
        The retrieved pointer of t_atoms in the dictionary has memory allocated and copied to it from within the function.
        You are responsible for freeing it with sysmem_freeptr().

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	argc	The address of a variable to hold the number of atoms in the array.
        @param	argv	The address of a variable to hold a pointer to the first atom in the array.
                        You should initialize this pointer to NULL prior to passing it to dictionary_copyatoms().
        @return			A Max error code.

        @see			dictionary_getatoms()
    */
    t_max_err dictionary_copyatoms(const t_dictionary* d, t_symbol* key, long* argc, t_atom** argv);


    /**
        Retrieve a #t_atomarray pointer from the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @return			A Max error code.
    */
    t_max_err dictionary_getatomarray(const t_dictionary* d, t_symbol* key, t_object** value);


    /**
        Retrieve a #t_dictionary pointer from the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @return			A Max error code.
    */
    t_max_err dictionary_getdictionary(const t_dictionary* d, t_symbol* key, t_object** value);

    /**
        Retrieve the address of a #t_atom array of in the dictionary within nested dictionaries.
        The address can index into nested dictionaries using the '::' operator.  For example,
        the key "field::subfield" will look for the value at key "field" and then look for the
        value "subfield" in the value found at "field".

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	ac		The number of return values
        @param	av		The return values
        @param	errstr	An error message if an error code was returned.  Optional, pass NULL if you don't want it.
        @return			A Max error code.
    */
    t_max_err dictionary_get_ex(t_dictionary* d, t_symbol* key, long* ac, t_atom** av, char* errstr);


    /**
        Retrieve a #t_object pointer from the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @return			A Max error code.
    */
    t_max_err dictionary_getobject(const t_dictionary* d, t_symbol* key, t_object** value);


    /**
        Test a key to set if the data stored with that key contains a #t_string object.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to test.
        @return			Returns true if the key contains a #t_string, otherwise returns false.
    */
    long dictionary_entryisstring(const t_dictionary* d, t_symbol* key);


    /**
        Test a key to set if the data stored with that key contains a #t_atomarray object.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to test.
        @return			Returns true if the key contains a #t_atomarray, otherwise returns false.
    */
    long dictionary_entryisatomarray(const t_dictionary* d, t_symbol* key);


    /**
        Test a key to set if the data stored with that key contains a #t_dictionary object.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to test.
        @return			Returns true if the key contains a #t_dictionary, otherwise returns false.
    */
    long dictionary_entryisdictionary(const t_dictionary* d, t_symbol* key);


    /**
        Test a key to set if it exists in the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to test.
        @return			Returns true if the key exists, otherwise returns false.
    */
    long dictionary_hasentry(const t_dictionary* d, t_symbol* key);


    /**
        Return the number of keys in a dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @return			The number of keys in the dictionary.
    */
    t_atom_long dictionary_getentrycount(const t_dictionary* d);


    /**
        Retrieve all of the key names stored in a dictionary.

        The numkeys and keys parameters should be initialized to zero.
        The dictionary_getkeys() method will allocate memory for the keys it returns.
        You are then responsible for freeing this memory using dictionary_freekeys().
        <em>You must use dictionary_freekeys(), not some other method for freeing the memory.</em>

        @ingroup 	dictionary
        @param		d		The dictionary instance.
        @param		numkeys	The address of a long where the number of keys retrieved will be set.
        @param		keys	The address of the first of an array #t_symbol pointers where the retrieved keys will be set.
        @return				A max error code.

        @remark		The following example demonstrates fetching all of the keys from a dictionary named 'd'
                     in order to iterate through each item stored in the dictionary.
        @code
        t_symbol	**keys = NULL;
        long		numkeys = 0;
        long		i;
        t_object	*anItem;

        dictionary_getkeys(d, &numkeys, &keys);
        for(i=0; i<numkeys; i++){
            // do something with the keys...
        }
        if(keys)
            dictionary_freekeys(d, numkeys, keys);
        @endcode

        @see 		dictionary_freekeys()
    */
    t_max_err dictionary_getkeys(const t_dictionary* d, long* numkeys, t_symbol*** keys);
    t_max_err dictionary_getkeys_ordered(const t_dictionary* d, long* numkeys, t_symbol*** keys);


    /**
        Free memory allocated by the dictionary_getkeys() method.

        @ingroup 	dictionary
        @param		d		The dictionary instance.
        @param		numkeys	The address of a long where the number of keys retrieved will be set.
        @param		keys	The address of the first of an array #t_symbol pointers where the retrieved keys will be set.

        @see 		dictionary_getkeys()
    */
    void dictionary_freekeys(t_dictionary* d, long numkeys, t_symbol** keys);


    /**
        Remove a value from the dictionary.
        This method will free the object in the dictionary.
        If freeing the object is inappropriate or undesirable, use dictionary_chuckentry() instead.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to delete.
        @return			A max error code.

        @see dictionary_chuckentry()
        @see dictionary_clear()
    */
    t_max_err dictionary_deleteentry(t_dictionary* d, t_symbol* key);


    /**
        Remove a value from the dictionary without freeing it.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to delete.
        @return			A max error code.

        @see dictionary_deleteentry()
    */
    t_max_err dictionary_chuckentry(t_dictionary* d, t_symbol* key);		// remove a value from the dictionary without deleting it


    /**
        Delete all values from a dictionary.
        This method will free the objects in the dictionary.
        If freeing the objects is inappropriate or undesirable then you should iterate through
        the dictionary and use dictionary_chuckentry() instead.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @return			A max error code.

        @see dictionary_getkeys()
        @see dictionary_chuckentry()
        @see dictionary_deleteentry()
    */
    t_max_err dictionary_clear(t_dictionary* d);



    t_dictionary* dictionary_clone(t_dictionary* d);
    t_max_err dictionary_clone_to_existing(const t_dictionary* d, t_dictionary* dc);
    t_max_err dictionary_copy_to_existing(const t_dictionary* d, t_dictionary* dc);
    t_max_err dictionary_merge_to_existing(const t_dictionary* d, t_dictionary* dc);


    // funall will pass the t_dictionary_entry pointer to the fun
    // use the methods below to access the fields

    /**
        Call the specified function for every entry in the dictionary.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	fun		The function to call, specified as function pointer cast to a Max #method.
        @param	arg		An argument that you would like to pass to the function being called.

        @remark			The dictionary_funall() method will call your function for every entry in the dictionary.
                        It will pass both a pointer to the #t_dictionary_entry, and any argument that you provide.
                        The following example shows a function that could be called by dictionary_funall().
        @code
        void my_function(t_dictionary_entry* entry, void* my_arg)
        {
            t_symbol	*key;
            t_atom		value;

            key = dictionary_entry_getkey(entry);
            dictionary_entry_getvalue(entry, &value);

            // do something with key, value, and my_arg...
        }
        @endcode
        @see dictionary_entry_getkey()
        @see dictionary_entry_getvalue()
    */
    void dictionary_funall(t_dictionary* d, method fun, void* arg);


    /**
        Given a #t_dictionary_entry*, return the key associated with that entry.

        @ingroup		dictionary
        @param	x		The dictionary entry.
        @return			The key associated with the entry.

        @see dictionary_entry_getvalue()
        @see dictionary_funall()
    */
    t_symbol* dictionary_entry_getkey(t_dictionary_entry* x);


    /**
        Given a #t_dictionary_entry*, return the value associated with that entry.

        @ingroup		dictionary
        @param	x		The dictionary entry.
        @param	value	The address of a #t_atom to which the value will be copied.

        @see dictionary_entry_getkey()
        @see dictionary_funall()
    */
    void dictionary_entry_getvalue(t_dictionary_entry* x, t_atom* value);

    /**
        Given a #t_dictionary_entry*, return the values associated with that entry.

        @ingroup		dictionary
        @param	x		The dictionary entry.
        @param	argc	The length of the returned #t_atom vector.
        @param	argv	The address of a #t_atom vector to which the values will be copied.

        @see dictionary_entry_getkey()
        @see dictionary_funall()
    */
    void dictionary_entry_getvalues(t_dictionary_entry* x, long* argc, t_atom** argv);



    /**
        Given 2 dictionaries, copy the keys unique to one of the dictionaries to the other dictionary.

        @ingroup			dictionary
        @param	d			A dictionary instance.  This will be the destination for any values that are copied.
        @param	copyfrom	A dictionary instance from which we will copy any values with unique keys.
        @return				A Max error code.

        @see				dictionary_copyentries()
    */
    t_max_err dictionary_copyunique(t_dictionary* d, t_dictionary* copyfrom);



    /**
        Retrieve a long integer from the dictionary.
        If the named key doesn't exist, then return a specified default value.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @param	def		The default value to return in the absence of the key existing in the dictionary.
        @return			A Max error code.

        @see			dictionary_getlong()
    */
    t_max_err dictionary_getdeflong(const t_dictionary* d, t_symbol* key, t_atom_long* value, t_atom_long def);


    /**
        Retrieve a double-precision float from the dictionary.
        If the named key doesn't exist, then return a specified default value.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @param	def		The default value to return in the absence of the key existing in the dictionary.
        @return			A Max error code.

        @see			dictionary_getfloat()
    */
    t_max_err dictionary_getdeffloat(const t_dictionary* d, t_symbol* key, double* value, double def);


    /**
        Retrieve a #t_symbol* from the dictionary.
        If the named key doesn't exist, then return a specified default value.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @param	def		The default value to return in the absence of the key existing in the dictionary.
        @return			A Max error code.

        @see			dictionary_getsym()
    */
    t_max_err dictionary_getdefsym(const t_dictionary* d, t_symbol* key, t_symbol** value, t_symbol* def);


    /**
        Retrieve a #t_atom* from the dictionary.
        If the named key doesn't exist, then return a specified default value.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @param	def		The default value to return in the absence of the key existing in the dictionary.
        @return			A Max error code.

        @see			dictionary_getatom()
    */
    t_max_err dictionary_getdefatom(const t_dictionary* d, t_symbol* key, t_atom* value, t_atom* def);


    /**
        Retrieve a C-string from the dictionary.
        If the named key doesn't exist, then return a specified default value.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	value	The address of variable to hold the value associated with the key.
        @param	def		The default value to return in the absence of the key existing in the dictionary.
        @return			A Max error code.

        @see			dictionary_getstring()
    */
    t_max_err dictionary_getdefstring(const t_dictionary* d, t_symbol* key, const char** value, char* def);


    /**
        Retrieve the address of a #t_atom array of in the dictionary.
        The retrieved pointer references the t_atoms in the dictionary.
        To fetch a copy of the t_atoms from the dictionary, use dictionary_copyatoms().
        If the named key doesn't exist, then return a default array of atoms, specified as a #t_atomarray*.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	argc	The address of a variable to hold the number of atoms in the array.
        @param	argv	The address of a variable to hold a pointer to the first atom in the array.
        @param	def		The default values specified as an instance of the #t_atomarray object.
        @return			A Max error code.

        @see			dictionary_getatoms()
        @see			dictionary_copydefatoms()
    */
    t_max_err dictionary_getdefatoms(t_dictionary* d, t_symbol* key, long* argc, t_atom** argv, t_atom* def);


    /**
        Retrieve copies of a #t_atom array in the dictionary.
        The retrieved pointer of t_atoms in the dictionary has memory allocated and copied to it from within the function.
        You are responsible for freeing it with sysmem_freeptr().
        If the named key doesn't exist, then copy a default array of atoms, specified as a #t_atomarray*.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	key		The key associated with the value to lookup.
        @param	argc	The address of a variable to hold the number of atoms in the array.
        @param	argv	The address of a variable to hold a pointer to the first atom in the array.
                        You should initialize this pointer to NULL prior to passing it to dictionary_copyatoms().
        @param	def		The default values specified as an instance of the #t_atomarray object.
        @return			A Max error code.

        @see			dictionary_getdefatoms()
        @see			dictionary_copyatoms()
    */
    t_max_err dictionary_copydefatoms(t_dictionary* d, t_symbol* key, long* argc, t_atom** argv, t_atom* def);



    /**
        Print the contents of a dictionary to the Max window.

        @ingroup		dictionary
        @param	d		The dictionary instance.
        @param	recurse	If non-zero, the dictionary will be recursively unravelled to the Max window.
                        Otherwise it will only print the top level.
        @param	console	If non-zero, the dictionary will be posted to the console rather than the Max window.
                        On the Mac you can view this using Console.app.
                        On Windows you can use the free DbgView program which can be downloaded from Microsoft.
        @return			A Max error code.
    */
    t_max_err dictionary_dump(t_dictionary* d, long recurse, long console);


    /**
        Copy specified entries from one dictionary to another.

        @ingroup		dictionary
        @param	src		The source dictionary from which to copy entries.
        @param	dst		The destination dictionary to which the entries will be copied.
        @param	keys	The address of the first of an array of #t_symbol* that specifies which keys to copy.
        @return			A Max error code.

        @see			dictionary_copyunique()
    */
    t_max_err dictionary_copyentries(t_dictionary* src, t_dictionary* dst, t_symbol** keys);


    /**
        Create a new dictionary populated with values using a combination of attribute and sprintf syntax.

        @ingroup		dictionary
        @param	fmt		An sprintf-style format string specifying key-value pairs with attribute nomenclature.
        @param	...		One or more arguments which are to be substituted into the format string.
        @return			A new dictionary instance.

        @remark			Max attribute syntax is used to define key-value pairs.  For example,
        @code
        "@key1 value @key2 another_value"
        @endcode

        @remark			One common use of this to create dictionary that represents an element of a patcher,
                        or even an entire patcher itself.  The example below creates a dictionary that can
                        be passed to a function like newobject_fromdictionary() to create a new object.
        @code
        t_dictionary* d;
        char text[4];

        strncpy_zero(text, "foo", 4);

        d = dictionary_sprintf("@maxclass comment @varname _name \
            @text \"%s\" @patching_rect %.2f %.2f %.2f %.2f \
            @fontsize %f @textcolor %f %f %f 1.0 \
            @fontname %s @bgcolor 0.001 0.001 0.001 0.",
            text, 20.0, 20.0, 200.0, 24.0,
            18, 0.9, 0.9, 0.9, "Arial");

        // do something with the dictionary here.

        object_free(d);
        @endcode

        @see			newobject_sprintf()
        @see			newobject_fromdictionary()
        @see			atom_setparse()
    */
    t_dictionary* dictionary_sprintf(const char* fmt, ...);


    /**
        Create a new object in a specified patcher with values using a combination of attribute and sprintf syntax.

        @ingroup		obj
        @param	patcher	An instance of a patcher object.
        @param	fmt		An sprintf-style format string specifying key-value pairs with attribute nomenclature.
        @param	...		One or more arguments which are to be substituted into the format string.
        @return			A pointer to the newly created object instance, or NULL if creation of the object fails.

        @remark			Max attribute syntax is used to define key-value pairs.  For example,
        @code
        "@key1 value @key2 another_value"
        @endcode

        @remark			The example below creates a new object that in a patcher whose
                        object pointer is stored in a variable called "aPatcher".
        @code
        t_object* my_comment;
        char text[4];

        strncpy_zero(text, "foo", 4);

        my_comment = newobject_sprintf(aPatcher, "@maxclass comment @varname _name \
            @text \"%s\" @patching_rect %.2f %.2f %.2f %.2f \
            @fontsize %f @textcolor %f %f %f 1.0 \
            @fontname %s @bgcolor 0.001 0.001 0.001 0.",
            text, 20.0, 20.0, 200.0, 24.0,
            18, 0.9, 0.9, 0.9, "Arial");
        @endcode

        @see			dictionary_sprintf()
        @see			newobject_fromdictionary()
        @see			atom_setparse()
    */
    t_object* newobject_sprintf(t_object* patcher, const char* fmt, ...);


    /**
        Create an object from the passed in text.
        The passed in text is in the same format as would be typed into an object box.
        It can be used for UI objects or text objects so this is the simplest way to create objects from C.

        @ingroup		obj
        @param	patcher	An instance of a patcher object.
        @param	text	The text as if typed into an object box.
        @return			A pointer to the newly created object instance, or NULL if creation of the object fails.

        @see newobject_sprintf()
    */
    t_object* newobject_fromboxtext(t_object* patcher, const char* text);


    /**
        Place a new object into a patcher.  The new object will be created based on a specification
        contained in a @ref dictionary.

        Create a new dictionary populated with values using a combination of attribute and sprintf syntax.

        @ingroup		obj
        @param	patcher	An instance of a patcher object.
        @param	d		A dictionary containing an object specification.
        @return			A pointer to the newly created object instance, or NULL if creation of the object fails.

        @remark			Max attribute syntax is used to define key-value pairs.  For example,
        @code
        "@key1 value @key2 another_value"
        @endcode

        @remark			The example below creates a new object that in a patcher whose
                        object pointer is stored in a variable called "aPatcher".
        @code
        t_dictionary* d;
        t_object* o;
        char text[4];

        strncpy_zero(text, "foo", 4);

        d = dictionary_sprintf("@maxclass comment @varname _name \
            @text \"%s\" @patching_rect %.2f %.2f %.2f %.2f \
            @fontsize %f @textcolor %f %f %f 1.0 \
            @fontname %s @bgcolor 0.001 0.001 0.001 0.",
            text, 20.0, 20.0, 200.0, 24.0,
            18, 0.9, 0.9, 0.9, "Arial");

        o = newobject_fromdictionary(aPatcher, d);
        @endcode

        @see			newobject_sprintf()
        @see			newobject_fromdictionary()
        @see			atom_setparse()
    */
    t_object* newobject_fromdictionary(t_object* patcher, t_dictionary* d);


    END_USING_C_LINKAGE

}} // namespace c74::max
