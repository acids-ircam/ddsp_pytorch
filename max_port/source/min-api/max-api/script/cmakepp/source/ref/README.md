# Reference Values


CMake has a couple of scopes every file has its own scope, every function has its own scope and you can only have write access to your `PARENT_SCOPE`.  So I searched for  simpler way to pass data throughout all scopes.  My solution is to use CMake's `get_property` and `set_property` functions.  They allow me to store and retrieve data in the `GLOBAL` scope - which is unique per execution of CMake. It is my RAM for cmake - It is cleared after the programm shuts down.

I wrapped the get_property and set_property commands in these shorter and simple functions:


### Function List


* [address_append](#address_append)
* [address_append_string](#address_append_string)
* [address_delete](#address_delete)
* [address_get](#address_get)
* [address_new](#address_new)
* [address_peek_back](#address_peek_back)
* [address_peek_front](#address_peek_front)
* [address_pop_back](#address_pop_back)
* [address_pop_front](#address_pop_front)
* [address_print](#address_print)
* [address_push_back](#address_push_back)
* [address_push_front](#address_push_front)
* [address_set](#address_set)
* [address_set_new](#address_set_new)
* [address_type_get](#address_type_get)
* [address_type_matches](#address_type_matches)
* [is_address](#is_address)

### Function Descriptions

## <a name="address_append"></a> `address_append`





## <a name="address_append_string"></a> `address_append_string`





## <a name="address_delete"></a> `address_delete`





## <a name="address_get"></a> `address_get`





## <a name="address_new"></a> `address_new`





## <a name="address_peek_back"></a> `address_peek_back`





## <a name="address_peek_front"></a> `address_peek_front`





## <a name="address_pop_back"></a> `address_pop_back`





## <a name="address_pop_front"></a> `address_pop_front`





## <a name="address_print"></a> `address_print`





## <a name="address_push_back"></a> `address_push_back`





## <a name="address_push_front"></a> `address_push_front`





## <a name="address_set"></a> `address_set`





## <a name="address_set_new"></a> `address_set_new`





## <a name="address_type_get"></a> `address_type_get`





## <a name="address_type_matches"></a> `address_type_matches`





## <a name="is_address"></a> `is_address`









```
address_new()   # returns a unique refernce (you can also choose any string)
address_set(ref [args ...]) # sets the reference to the list of arguments
address_get(ref) # returns the data stored in <ref> 

# some more specialized functions
# which might be faster in special cases
address_set_new([args ...])    # creates, returns a <ref> which is set to <args>
address_print(<ref>)      # prints the ref
is_address(<ref>)      # returns true iff the ref is valid
address_type_matches(<ref> <type>)  # returns true iff ref is type
address_type_get(<ref>)      # returns the (if any) of the ref       
address_delete(<ref>)     # later: frees the specified ref
address_append(<ref> [args ...])# appends the specified args to the <ref>'s value
address_append_string(<ref> <string>) # appends <string> to <ref>'s value
```

*Example*:
```
 # create a ref
 address_new()
 ans(ref)
 assert(ref)
 
 # set the value of a ref
 address_set(${ref} "hello world")

# retrieve a value by dereferencing
address_get(${ref})
ans(val)
assert(${val} STREQUAL "hello world")

# without generating the ref:  
address_set("my_ref_name" "hello world")
address_get("my_ref_name")
ans(val)
assert(${val} STREQUAL "hello world")
```
