/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {

    BEGIN_USING_C_LINKAGE

    /**	Register a #t_dictionary with the dictionary passing system and map it to a unique name.

        @ingroup dictobj
        @param		d		A valid dictionary object.
        @param		name	The address of a #t_symbol pointer to the name you would like mapped to this dictionary.
                            If the t_symbol pointer has a NULL value then a unique name will be generated and filled-in
                            upon return.
        @return				The dictionary mapped to the specified name.
    **/
    t_dictionary* dictobj_register(t_dictionary* d, t_symbol** name);


    /**	Unregister a #t_dictionary with the dictionary passing system.
        Generally speaking you should not need to call this method.
        Calling object_free() on the #t_dictionary automatically unregisters it.

        @ingroup dictobj
        @param		d		A valid dictionary object.
        @return				A Max error code.
    **/
    t_max_err dictobj_unregister(t_dictionary* d);


    /**	Find the #t_dictionary for a given name, and return a <i>copy</i> of that dictionary
        When you are done, do <i>not</i> call dictobj_release() on the dictionary,
        because you are working on a copy rather than on a retained pointer.

        @ingroup dictobj
        @param		name	The name associated with the dictionary for which you wish to obtain a copy.
        @return				The dictionary cloned from the existing dictionary.
                            Returns NULL if no dictionary is associated with name.
        @see				#dictobj_findregistered_retain()
    **/
    t_dictionary* dictobj_findregistered_clone(const t_symbol* name);


    /**	Find the #t_dictionary for a given name, return a pointer to that #t_dictionary, and increment its reference count.
        When you are done you should call dictobj_release() on the dictionary.

        @ingroup dictobj
        @param		name	The name associated with the dictionary for which you wish to obtain a pointer.
        @return				A pointer to the dictionary associated with name.
                            Returns NULL if no dictionary is associated with name.
        @see				#dictobj_release()
        @see				#dictobj_findregistered_clone()
    **/
    t_dictionary* dictobj_findregistered_retain(const t_symbol* name);


    /**	For a #t_dictionary/name that was previously retained with dictobj_findregistered_retain(),
        release it (decrement its reference count).

        @ingroup dictobj
        @param		d		A valid dictionary object retained by dictobj_findregistered_retain().
        @return				A Max error code.
        @see				#dictobj_findregistered_retain()
    **/
    t_max_err dictobj_release(t_dictionary* d);


    /**	Find the name associated with a given #t_dictionary.

        @ingroup	dictobj
        @param		d		A dictionary, whose name you wish to determine.
        @return				The symbol associated with the dictionary, or NULL if the dictionary is not registered.
        @see				#dictobj_register()
    **/
    t_symbol* dictobj_namefromptr(t_dictionary* d);


    /**	Send atoms to an outlet in your Max object, handling complex datatypes that may be present in those atoms.
        This is particularly when sending the contents of a dictionary entry out of an outlet as in the following example code.

        @code
        long		ac = 0;
        t_atom		*av = NULL;
        t_max_err	err;

        err = dictionary_copyatoms(d, key, &ac, &av);
        if (!err && ac && av) {
            // handles singles, lists, symbols, atomarrays, dictionaries, etc.
            dictobj_outlet_atoms(x->outlets[i],ac,av);
        }

        if (av)
            sysmem_freeptr(av);
        @endcode

        @ingroup	dictobj
        @param		out		The outlet through which the atoms should be sent.
        @param		argc	The count of atoms in argv.
        @param		argv	Pointer to the first of an array of atoms to send to the outlet.
    **/
    void dictobj_outlet_atoms(void* out, long argc, t_atom* argv);


    /**	Ensure that an atom is safe for passing.
        Atoms are allowed to be #A_LONG, #A_FLOAT, or #A_SYM, but not #A_OBJ.
        If the atom is an #A_OBJ, it will be converted into something that will be safe to pass.

        @ingroup dictobj
        @param		a	An atom to check, and potentially modify, to ensure safety in the dictionary-passing system.
        @return			If the atom was changed then 1 is returned.  Otherwise 0 is returned.
    **/
    long dictobj_atom_safety(t_atom* a);


    enum {
        DICTOBJ_ATOM_FLAGS_DEFAULT = 0,	///< default
        DICTOBJ_ATOM_FLAGS_REGISTER		///< dictionary atoms should be registered/retained
    };


    /**	Ensure that an atom is safe for passing.
        Atoms are allowed to be #A_LONG, #A_FLOAT, or #A_SYM, but not #A_OBJ.
        If the atom is an #A_OBJ, it will be converted into something that will be safe to pass.

        @ingroup dictobj
        @param		a		An atom to check, and potentially modify, to ensure safety in the dictionary-passing system.
        @param		flags	Pass DICTOBJ_ATOM_FLAGS_REGISTER to have dictionary atoms registered/retained.
        @return				If the atom was changed then 1 is returned.  Otherwise 0 is returned.
     **/
    long dictobj_atom_safety_flags(t_atom* a, long flags);


    void dictobj_atom_release(t_atom* a);


    /**	Validate the contents of a #t_dictionary against a second #t_dictionary containing a schema.

        The schema dictionary contains keys and values, like any dictionary.
        dictobj_validate() checks to make sure that all keys in the schema dictionary are present in the candidate dictionary.
        If the keys are all present then the candidate passes and the function returns true.
        Otherwise the the candidate fails the validation and the function returns false.

        Generally speaking, the schema dictionary with contain values with the symbol "*", indicating a wildcard,
        and thus only the key is used to validate the dictionary (all values match the wildcard).
        However, if the schema dictionary contains non-wildcard values for any of its keys, those keys in the
        candidate dictionary must also contain matching values in order for the candidate to successfully validate.

        An example of this in action is the dict.route object in Max, which simply wraps this function.

        @ingroup dictobj
        @param		schema		The dictionary against which to validate candidate.
        @param		candidate	A dictionary to test against the schema.
        @return					Returns true if the candidate validates against the schema, otherwise returns false.
        @see					#dictobj_dictionarytoatoms()
    **/
    long dictobj_validate(const t_dictionary* schema, const t_dictionary* candidate);


    /**	Convert a C-string of @ref using_dictobj_syntax into a C-string of JSON.

        @ingroup dictobj
        @param		jsonsize	The address of a variable to be filled-in with the number of chars in json upon return.
        @param		json		The address of a char pointer to point to the JSON C-string upon return.
                                Should be initialized to NULL.
                                You are responsible for freeing the string with sysmem_freeptr() when you are done with it.
        @param		str			A NULL-terminated C-string containing @ref using_dictobj_syntax .
        @return					A Max error code.
        @see					#dictobj_dictionarytoatoms()
    **/
    t_max_err dictobj_jsonfromstring(long* jsonsize, char** json, const char* str);


    /**	Create a new #t_dictionary from @ref using_dictobj_syntax which is passed in as a C-string.

        @ingroup dictobj
        @param		d		The address of a dictionary variable, which will hold a pointer to
                            the new dictionary upon return.  Should be initialized to NULL.
        @param		str		A NULL-terminated C-string containing @ref using_dictobj_syntax .
        @param		str_is_already_json .
        @param		errorstring .
        @return				A Max error code.
        @see				#dictobj_dictionarytoatoms()
    **/
    t_max_err dictobj_dictionaryfromstring(t_dictionary** d, const char* str, int str_is_already_json, char* errorstring);


    /**	Create a new #t_dictionary from @ref using_dictobj_syntax which is passed in as an array of atoms.
        Unlike many #t_dictionary calls to create dictionaries, this function does not take over ownership of the atoms you pass in.

        @ingroup dictobj
        @param		d		The address of a dictionary variable, which will hold a pointer to
                            the new dictionary upon return.  Should be initialized to NULL.
        @param		argc	The number of atoms in argv.
        @param		argv	Pointer to the first of an array of atoms to be interpreted as
                            @ref using_dictobj_syntax .
        @return		A Max error code.
        @see		#dictobj_dictionaryfromatoms_extended() #dictobj_dictionarytoatoms()
    **/
    t_max_err dictobj_dictionaryfromatoms(t_dictionary** d, const long argc, const t_atom* argv);


    /**	Create a new #t_dictionary from from an array of atoms that use Max dictionary syntax, JSON, or compressed JSON.
        This function is the C analog to the dict.deserialize object in Max.
        Unlike many #t_dictionary calls to create dictionaries, this function does not take over ownership of the atoms you pass in.

         @ingroup dictobj
         @param		d		The address of a dictionary variable, which will hold a pointer to
                            the new dictionary upon return.  Should be initialized to NULL.
         @param		msg		Ignored.
         @param		argc	The number of atoms in argv.
         @param		argv	Pointer to the first of an array of atoms to be interpreted as @ref using_dictobj_syntax , JSON, or compressed JSON.
         @return	A Max error code.
         @see		#dictobj_dictionaryfromatoms() #dictobj_dictionaryfromstring()
     **/
    t_max_err dictobj_dictionaryfromatoms_extended(t_dictionary** d, const t_symbol* msg, long argc, const t_atom* argv);


    /**	Serialize the contents of a #t_dictionary into @ref using_dictobj_syntax .

        @ingroup dictobj
        @param		d		The dictionary to serialize.
        @param		argc	The address of a variable to hold the number of atoms allocated upon return.
        @param		argv	The address of a t_atom pointer which will point to the first atom
                            (of an array of argc atoms) upon return.

        @return		A Max error code.
        @see		#dictobj_dictionaryfromatoms()
    **/
    t_max_err dictobj_dictionarytoatoms(const t_dictionary* d, long* argc, t_atom** argv);


    /**	Given a complex key (one that includes potential heirarchy or array-member access),
        return the actual key and the dictionary in which the key should be referenced.

        @ingroup	dictobj
        @param		x			Your calling object.  If there is an error this will be used by the internal call to object_error().
        @param		d			The dictionary you are querying.
        @param		akey		The complex key specifying the query.
        @param		create		If true, create the intermediate dictionaries in the hierarchy specified in akey.
        @param		targetdict	Returns the t_dictionary that for the (sub)dictionary specified by akey.
        @param		targetkey	Returns the name of the key in targetdict that to which akey is referencing.
        @param		index		Returns the index requested if array-member access is specified.  Pass NULL if you are not interested in this.

        @return		A Max error code.
     */
    t_max_err dictobj_key_parse(t_object* x, t_dictionary* d, t_atom* akey, t_bool create, t_dictionary** targetdict, t_symbol** targetkey, t_int32 *index);


    END_USING_C_LINKAGE

}} // namespace c74::max
