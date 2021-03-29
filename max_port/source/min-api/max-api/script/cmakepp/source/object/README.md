## Objects 

Objects are an extension of the maps. They add inheritance, member calls and custom member operations to the concept of maps. I split up maps and objects because objects are a lot slower (something like 2x-3x slower) and if you do not need objects you should not use them (handling 1000s of maps is already slow enough). The reason for the performance loss is the number of function calls needed to get the correct virtual function/property value.


## Functions and Datatypes

These are the basic functions which are available (there are more which all use these basic ones): 

* Basic Object functions - Functions on which all other object functions base
  - `<type> := <cmake function>` a type is represented by a globally unique cmake function.  This function acts as the constructor for the type. In this constructor you define inheritance, properties and member functions. The type may only be defined once.  So make sure the function name is and stays unique ie. is not overwritten somewhere else in the code.
  - `<object> := <ref>` an object is the instance of a type. 
  - `<member> := <key>` a string which identifies the member of an object
  - `obj_new(<type>):<object>` creates the instance of a type. calls the constructor function specified by type and returns an object instance
  - `obj_delete(<object>):<void>` deletes an object instance. You MAY call this. If you do the desctructor of an object is invoked. This function is defined for completeness sake and can only be implemented if CMake script changes. So don't worry about this ;).
  - `obj_set(<object> <member> <value:<any...>>):<any>`  sets the object's  property identified by `<member>` to the specified value.  *the default behaviour can be seen in `obj_default_setter(...)` and MAY be overwridden by using `obj_declare_setter(...)`*
  - `obj_get(<object> <member>):<any>` gets the value of the object's property identified by `<member>` *the default behaviour MAY be overwridden b using `obj_declare_getter`*  
  - `obj_has(<object> <member>):<bool>` returns true iff the object has a property called `<member>`
  - `obj_keys(<object>):<member...>` returns the list of available members
  - `obj_member_call(<object> <member> <args:<any...>>):<any>` calls the specified member with the specified arguments and returns the result of the operation
* Most `obj_*` functions are also available in the shorter `this-form` so `obj_get(<this:<object>> ...)` can also be used inside a member function or constructor by using the function `this_get(...)`.  This is achieved by forwarding the call from `this_get(...)` to `obj_get(${this} ...)`

## Overriding default behaviour

As is possible in JavaScript an Python I let you override default object operations by specifying custom member functions. You may set the following hidden object fields to the name of a function which implements the correct interface. You can see the default implementations by looking at the `obj_default_*` functions.  To ensure ease of use I provided functions which help you correctly override object behaviour.  

* reserved hidden object fields
  * `__setter__ : (<this> <member> <any...>):<any>` override the set operation. Return value may be anything. the default is void
  * `__getter__ : (<this> <member>):<any>` override the get operation. expects the return value to be the value of the object's property identified by `<member>`
  * `__get_keys__ : (<this>):<member ...>` override the operation which returns the list of available keys.  It is expected that all keys returned will are valid properties (they exist).
  * `__has__ : (<this> <member>):<bool>` overrides the has operation. MUST return true iff the object has a property called `<member>`
  * `__member_call__ : (<this> <member> <args:<any...>>):<return value:<any>>` this operation is invoked when `obj_member_call(...)` is called (and thus also when `call, rcall, etc` is called) overriding this function allows you to dispatch a call operation to the object member identified by `<member>` with the specified `args` it should return the result of the specified operation. The `this` variable is always set to the object instance current instance.
  * `__cast__` 
* helper functions
  * `obj_declare_getter()`
  * `obj_declare_setter()`
  * `obj_declare_call()`
  * `obj_declare_member_call()`  
  * `obj_declare_get_keys()`
  * `obj_declare_has_key()` 
  * `obj_declare_cast()` 
  

```
new([Constructor]) returns a ref to a object
obj_get(obj)
obj_set(obj)
obj_has(obj)
obj_owns(obj)
obj_keys(obj)
obj_ownedkeys(obj)
obj_call(obj)
obj_member_call(obj key [args])
obj_delete(obj)
```



## Example

This is how you define prototypes and instanciate objects.  

The syntax seems a bit strange and could be made much easier with a minor change to CMake... Go Cmake Gods give me true macro power! (allow to define a macro which can call function() and another which contains endfunction()) /Rant



```
function(BaseType)
  # initialize a field
  this_set(accu 0)
  # declare a functions which adds a value to the accumulator of this object
  proto_declarefunction(accuAdd)
  function(${accuAdd} b)
    this_get(accu)
    math_eval("${accu} + ${b}")
    ans(accu)
    this_set(accu "${accu}")
    call(this.printAccu())
    return(${accu})
  endfunction()

  proto_declarefunction(printAccu)
  function(${printAccu})
    this_get(accu)
    message("value of accu: ${accu}")
  endfunction()
endfunction()
function(MyType)
  # inherit another type
  this_inherit(BaseType)
  # create a subtract from accu function
  proto_declarefunction(accuSub)
  function(${accuSub} b)
    this_get(accu)
    math_eval("${accu} - ${b}")
    this_set(accu "${accu}")
    call(this.printAccu())
    return(${accu})
  endfunction()

endfunction()

new(MyType)
ans(myobj)
rcall(result = myobj.add(3))
# result == 3, output 3
rcall(result = myobj.sub(2))
# result == 1, output 1
```

## Special hidden Fields
```
__type__    # contains the name of the constructor function
__proto__   # contains the prototype for this object
__getter__    # contains a function (obj, key)-> value 
__setter__    # contains a function (obj, key,value) 
__call__    # contains a function (obj [arg ...])
__callmember__  # contains a function (obj key [arg ..])
__not_found__   # gets called by the default __getter__ when a field is not found
__to_string__ # contains a function which returns a string representation for the object
__destruct__  # a function that is called when the object is destroyed
```
