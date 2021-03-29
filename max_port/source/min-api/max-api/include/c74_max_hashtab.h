/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_max_linklist.h"

namespace c74 {
namespace max {

    /** A hashtab entry. This struct is provided for debugging convenience,
        but should be considered opaque and is subject to change without notice.

        @ingroup hashtab
        @see t_hashtab
    */
    typedef struct _hashtab_entry
    {
        t_object ob;
        t_symbol *key;
        t_object *value;
        long flags;
        struct _hashtab *parent;
    } t_hashtab_entry;

    /** The hashtab object. This struct is provided for debugging convenience,
        but should be considered opaque and is subject to change without notice.

        @ingroup hashtab
        @see t_hashtab
    */
    typedef struct _hashtab
    {
        t_object ob;
        long slotcount;
        t_linklist **slots;
        long readonly;
        long flags;
        void *reserved;
    } t_hashtab;


    BEGIN_USING_C_LINKAGE

    /**
        Create a new hashtab object.
        You can free the hashtab by calling object_free() on the hashtab's pointer,
        or by using hashtab_chuck().

        @ingroup hashtab
        @param	slotcount	The number of slots in the hash table.  Prime numbers typically work well.
                            Pass 0 to get the default size.
        @return				Pointer to the new hashtab object.

        @see				object_free()
        @see				hashtab_chuck()
    */
    t_hashtab* hashtab_new(long slotcount);


    /**
        Store an item in a hashtab with an associated key.

        @ingroup hashtab

        @param	x		The hashtab instance.
        @param	key		The key in the hashtab with which to associate the value.
        @param	val		The value to store.

        @return			A Max error code.
        @see			hashtab_lookup(), hashtab_storesafe(), hashtab_storelong(), hashtab_storesym()
    */
    t_max_err hashtab_store(t_hashtab* x, t_symbol* key, t_object* val);

    /**
        Store a t_atom_long value in a hashtab with an associated key.

        @ingroup hashtab

        @param	x		The hashtab instance.
        @param	key		The key in the hashtab with which to associate the value.
        @param	val		The t_atom_long value to store.

        @return			A Max error code.
        @see			hashtab_lookuplong(), hashtab_store(), hashtab_storesafe(), hashtab_storesym()
    */
    t_max_err hashtab_storelong(t_hashtab* x, t_symbol* key, t_atom_long val);

    /**
        Store a t_symbol value in a hashtab with an associated key.

        @ingroup hashtab

        @param	x		The hashtab instance.
        @param	key		The key in the hashtab with which to associate the value.
        @param	val		The t_symbol pointer to store.

        @return			A Max error code.
        @see			hashtab_lookupsym(), hashtab_store(), hashtab_storesafe(), hashtab_storelong()
    */
    t_max_err hashtab_storesym(t_hashtab* x, t_symbol* key, t_symbol* val);


    /**	Store an item in a hashtab with an associated key.
        The difference between hashtab_store_safe() and hashtab_store() is what happens in the event of a collision in the hash table.
        The normal hashtab_store() function will free the existing value at the collision location with sysmem_freeptr() and then replaces it.
        This version doesn't try to free the existing value at the collision location, but instead just over-writes it.

        @ingroup		hashtab
        @param	x		The hashtab instance.
        @param	key		The key in the hashtab with which to associate the value.
        @param	val		The value to store.
        @return			A Max error code.
        @see			hashtab_store()
    */
    t_max_err hashtab_store_safe(t_hashtab* x, t_symbol* key, t_object* val);


    /**	Store an item in a hashtab with an associated key and also flags that define the behavior of the item.
        The hashtab_store() method is the same as calling this method with the default (0) flags.

        @ingroup		hashtab
        @param	x		The hashtab instance.
        @param	key		The key in the hashtab with which to associate the value.
        @param	val		The value to store.
        @param	flags	One of the values listed in #e_max_datastore_flags.
        @return			A Max error code.
        @see			hashtab_store()
    */
    t_max_err hashtab_storeflags(t_hashtab* x, t_symbol* key, t_object* val, long flags);


    /**
        Return an item stored in a hashtab with the specified key.

        @ingroup hashtab

        @param	x		The hashtab instance.
        @param	key		The key in the hashtab to fetch.
        @param	val		The address of a pointer to which the fetched value will be assigned.

        @return			A Max error code.
        @see			hashtab_store(), hashtab_lookuplong(), hashtab_lookupsym()
    */
    t_max_err hashtab_lookup(t_hashtab* x, t_symbol* key, t_object** val);

    /**
        Return a t_atom_long value stored in a hashtab with the specified key.

        @ingroup hashtab

        @param	x		The hashtab instance.
        @param	key		The key in the hashtab to fetch.
        @param	val		A pointer to a t_atom_long to which the fetched value will be assigned.

        @return			A Max error code.
        @see			hashtab_storelong(), hashtab_lookup(), hashtab_lookupsym()
    */
    t_max_err hashtab_lookuplong(t_hashtab* x, t_symbol* key, t_atom_long* val);

    /**
        Return a t_symbol value stored in a hashtab with the specified key.

        @ingroup hashtab

        @param	x		The hashtab instance.
        @param	key		The key in the hashtab to fetch.
        @param	val		A pointer to the address of a t_symbol to which the fetched value will be assigned.

        @return			A Max error code.
        @see			hashtab_storesym(), hashtab_lookup(), hashtab_lookuplong()
    */
    t_max_err hashtab_lookupsym(t_hashtab* x, t_symbol* key, t_symbol** val);



    /**	Return an item stored in a hashtab with the specified key, also returning the items flags.
        @ingroup		hashtab
        @param	x		The hashtab instance.
        @param	key		The key in the hashtab to fetch.
        @param	val		The address of a pointer to which the fetched value will be assigned.
        @param	flags	The address of a value to which the fetched flags will be assigned.
        @return			A Max error code.
        @see			hashtab_lookup()
        @see			hashtab_store_flags()
    */
    t_max_err hashtab_lookupflags(t_hashtab* x, t_symbol* key, t_object** val, long* flags);


    /**
        Remove an item from a hashtab associated with the specified key and free it.

        The hashtab can contain a variety of different types of data.
        By default, the hashtab assumes that all items are max objects with a valid
        #t_object header.  Thus by default, it frees items by calling object_free() on them.

        You can alter the hashtab's notion of what it contains by using the
        hashtab_flags() method.

        If you wish to remove an item from the hashtab and free it yourself, then you
        should use hashtab_chuckkey().

        @ingroup hashtab

        @param	x		The hashtab instance.
        @param	key		The key of the item to delete.
        @return			A Max error code.

        @see			hashtab_chuckkey()
        @see			hashtab_clear()
        @see			hashtab_flags()
    */
    t_max_err hashtab_delete(t_hashtab* x, t_symbol* key);


    /**
        Delete all items stored in a hashtab.
        This is the equivalent of calling hashtab_delete() on every item in a hashtab.

        @ingroup	hashtab
        @return		A max error code.
        @see		hashtab_flags()
        @see		hashtab_delete()
    */
    t_max_err hashtab_clear(t_hashtab* x);


    /**
        Remove an item from a hashtab associated with a given key.

        You are responsible for freeing any memory associated with the item that is
        removed from the hashtab.

        @ingroup hashtab

        @param	x		The hashtab instance.
        @param	key		The key of the item to delete.
        @return			A Max error code.

        @see			hashtab_delete
    */
    t_max_err hashtab_chuckkey(t_hashtab* x, t_symbol* key);


    /**
        Free a hashtab, but don't free the items it contains.

        The hashtab can contain a variety of different types of data.
        By default, the hashtab assumes that all items are max objects with a valid
        #t_object header.

        You can alter the hashtab's notion of what it contains by using the
        hashtab_flags() method.

        When you free the hashtab by calling object_free() it then tries to free all of the items it contains.
        If the hashtab is storing a custom type of data, or should otherwise not free the data it contains,
        then call hashtab_chuck() to free the object instead of object_free().

        @ingroup	hashtab
        @param	x	The hashtab object to be freed.
        @return		A max error code.
        @see		object_free
    */
    t_max_err hashtab_chuck(t_hashtab* x);


    /**
        Search the hash table for the first item meeting defined criteria.
        The items in the hashtab are iteratively processed, calling a specified comparison function on each
        until the comparison function returns true.

        @ingroup hashtab
        @param	x		The hashtab instance.
        @param	o		The address to pointer that will be set with the matching item.
        @param	cmpfn	The function used to determine a match in the list.
        @param	cmpdata	An argument to be passed to the #t_cmpfn.
                        This will be passed as the second of the two args to the #t_cmpfn.
                        The first arg will be the hashtab item at each iteration in the list.
        @return			A max error code.

        @see			linklist_findfirst()
        @see			t_cmpfn
    */
    t_max_err hashtab_findfirst(t_hashtab* x, void** o, long cmpfn(void* , void* ), void* cmpdata);


    /**
        Call the named message on every object in the hashtab.
        The hashtab_methodall() function requires that all items in the hashtab are
        object instances with a valid #t_object header.

        @ingroup hashtab
        @param	x	The hashtab instance.
        @param	s	The name of the message to send to the objects.
        @param	...	Any arguments to be sent with the message.
        @return		A max error code.

        @remark		Internally, this function uses object_method(), meaning that no errors will be
                    posted if the message name does not exist for the object.  It also means that
                    messages sent methods with #A_GIMME definitions will need to be given a symbol
                    argument prior to the argc and argv array information.
    */
    t_max_err hashtab_methodall(t_hashtab* x, t_symbol* s, ...);


    /**
        Call the specified function for every item in the hashtab.

        @ingroup hashtab
        @param	x		The hashtab instance.
        @param	fun		The function to call, specified as function pointer cast to a Max #method.
        @param	arg		An argument that you would like to pass to the function being called.
        @return			A max error code.

        @remark			The hashtab_funall() method will call your function for every item in the list.
                        It will pass both a pointer to the item in the list, and any argument that you
                        provide.  The following example shows a function that could be called by hashtab_funall().
        @code
        void myFun(t_hashtab_entry *e, void* myArg)
        {
            if (e->key && e->value) {
                // do something with e->key, e->value, and myArg here as appropriate
            }
        }
        @endcode
    */
    t_max_err hashtab_funall(t_hashtab* x, method fun, void* arg);



    /**
        Return the number of items stored in a hashtab.

        @ingroup	hashtab
        @param	x	The hashtab instance.
        @return		The number of items in the hash table.
    */
    t_atom_long hashtab_getsize(t_hashtab* x);


    /**
        Post a hashtab's statistics to the Max window.

        @ingroup	hashtab
        @param	x	The hashtab instance.
    */
    void hashtab_print(t_hashtab* x);


    /**
        Set the hashtab's readonly bit.

        By default the readonly bit is 0, indicating that it is threadsafe for both reading and writing.
        Setting the readonly bit to 1 will disable the hashtab's theadsafety mechanism, increasing
        performance but at the expense of threadsafe operation.
        Unless you can guarantee the threading context for a hashtab's use, you should leave this set to 0.

        @ingroup hashtab
        @param	x			The hashtab instance.
        @param	readonly	A 1 or 0 for setting the readonly bit.
    */
    void hashtab_readonly(t_hashtab* x, long readonly);


    /**
        Set the hashtab's datastore flags.  The available flags are enumerated in #e_max_datastore_flags.
        These flags control the behavior of the hashtab, particularly when removing items from the list
        using functions such as hashtab_clear(), hashtab_delete(), or when freeing the hashtab itself.

        @ingroup hashtab
        @param	x		The hashtab instance.
        @param	flags	A valid value from the #e_max_datastore_flags.  The default is #OBJ_FLAG_OBJ.
    */
    void hashtab_flags(t_hashtab* x, long flags);


    /**
        Get the hashtab's datastore flags.

        @ingroup hashtab
        @param	x	The hashtab instance.
        @return		The current state of the hashtab flags as enumerated in #e_max_datastore_flags.
    */
    t_atom_long hashtab_getflags(t_hashtab* x);


    /**	Change the flags for an item stored in the hashtab with a given key.
        @ingroup		hashtab
        @param	x		The hashtab instance.
        @param	key		The key in the hashtab whose flags will be changed.
        @param	flags	One of the values listed in #e_max_datastore_flags.
        @return			A Max error code.
        @see			hashtab_store_flags()
    */
    t_max_err hashtab_keyflags(t_hashtab* x, t_symbol* key, long flags);


    /**	Retrieve the flags for an item stored in the hashtab with a given key.
        @ingroup		hashtab
        @param	x		The hashtab instance.
        @param	key		The key in the hashtab whose flags will be returned.
        @return			The flags for the given key.
        @see			hashtab_store_flags()
    */
    t_atom_long hashtab_getkeyflags(t_hashtab* x, t_symbol* key);


    /**
        Retrieve all of the keys stored in a hashtab.

        If the kc and kv parameters are properly initialized to zero, then hashtab_getkeys() will allocate memory
        for the keys it returns.  You are then responsible for freeing this memory using sysmem_freeptr().

        @ingroup hashtab
        @param		x	The hashtab instance.
        @param		kc	The address of a long where the number of keys retrieved will be set.
        @param		kv	The address of the first of an array #t_symbol pointers where the retrieved keys will be set.
        @return		A max error code.

        @remark		The following example demonstrates fetching all of the keys from a hashtab in order to iterate through
                    each item stored in the hashtab.
        @code
        t_symbol	**keys = NULL;
        long		numKeys = 0;
        long		i;
        t_object	*anItem;

        hashtab_getkeys(aHashtab, &numKeys, &keys);
        for(i=0; i<numKeys; i++){
            hashtab_lookup(aHashtab, keys[i], &anItem);
            // Do something with anItem here...
        }
        if(keys)
            sysmem_freeptr(keys);
        @endcode
    */
    t_max_err hashtab_getkeys(t_hashtab* x, long* kc, t_symbol*** kv);



    END_USING_C_LINKAGE

}} // namespace c74::max
