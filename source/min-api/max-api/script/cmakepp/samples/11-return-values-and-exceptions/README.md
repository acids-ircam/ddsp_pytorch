# Dynamic Calls, Return Values and Exceptions


## Motivation

`CMake` does not provide a mechanism for function values or handling exceptional cases.  For the normal procedural style of `CMakeLists` this of course has little to no bearing because a simple `if-statement` will probably be sufficient to check if something went wrong and you know the signatures of the functions you use.  But as soon as you want to create something languistically more complex than the trivial case you run into problems where `CMake` leaves you high and dry.  It does not support calling a function whose name is not known at runtime and even if it did you would not necessarily know the signature so you could not say which values are input and which are output values.  

Even though people might say that `CMake` is overly simple and not much more than a markup language it still provides the basic functionality needed to create a complex and easy to use programming language:

* functions with child scope
* overwriteable functions
* parent scope modification `set(<var> <any>... PARENT_SCOPE)`
* `include` script files and 
* `file(WRITE)` 

Using these built in features `CMake` can be coerced into a more powerful and easy to use language (without needing to modify the interpreter itself) which provides:

* runtime function calls (variable name functions or dynamic call)
* function return values (a convention based approach)
* exceptions and exception handlers (`throw` and `catch`)
* ...

Since these subjects belong together intimately I want to describe them in a single Document.

## Dynamic Call

Consider the following trivial case (which in no way warrents to be treated with such complex structures as I am about to introduce):

You want to create a target which can be either an `executable` or a `library` depending on a configuration flag:

```cmake
cmake_minimum_required(VERSION 2.8.7) 
project(my_project)
set(sources file1.h file1.cpp)
set(name my_target)
if("${target_type}" STREQUAL "executable")
    add_executable("${name}" ${sources})
elseif("${target_type}" STREQUAL "library")
    add_library("${name}" ${sources})
endif()
```

If you invoke the `CMakeLists.txt` above with the command line option `-Dtarget_type=exectuable` and exectuable will be created and if the option `-Dtarget_type=library` is provided a library will be created.

However what would happen if you had much more cases than these two (custom targets, custom functions which define multiple targets etc.): You would end up with a huge `if-statement` which would have to be modified every time a new options is added.  This is a code smell: Brittle Desing.  So what you actually want instead of the large `if-statement` is the following

```cmake
...
add_${target_type}("${name}" ${sources})
...
```



Of course: This is invalid `CMake` syntax  because the function name needs to be a valid cmake identifier (which matches the regex `[A-Za-z_][A-Za-z0-9_]*`.  And even then would cmake not evaluate the variable expression `${target_type}` if it is not inside a command invocation.

So the first thing you need is a way to dynamically invoke cmake and this is what I came up with:
```cmake
cmake_minimum_required(VERSION 2.8.7) 
include(cmakepp.cmake) ## required for the functionality
...
eval_cmake(${target_type}("\${name}" \${sources})) # the backslash causes the evaluation to occur later else the variables would be evaluated directly which in this case does not matter
...
```

Now you are able to dynamically invoke any function whose name is inside a variable.

**Note**

`eval*` is slow.  It has overhead of about takes about a millisecond on a `i7` and a SSD on Windows.  It is so slow because what actually happens is that a temporary file is create with the dynamic cmake code inside which is directly included after it was written (see `eval` function).  The file operation causes  `eval*` to be slow.  This of course is only a problem if you choose to eval _alot_.

## Return Values

This was all very simple and only works if you do not want to have any output.  `eval`, `eval_cmake` etc are functions which means that they have a scope and no matter how hard you try you can never have a fast reliable solution which sets `PARENT_SCOPE` variables of a child function invocation. I have a tried a lot of different strategies and came to the conclusion that if you want to return something you should not use output variables because the cause pain and despair.  

The solution I chose, and which works very well is that I introduce a convention for return values.  After a function call, if it returns a value only a variable called `__ans` is modified in the `PARENT_SCOPE` e.g.:

```cmake
## the function
function(list_length list_ref)
    list(LENGTH "${list_ref}" length)
    set(__ans ${length} PARENT_SCOPE)
endfunction()

set(my_list 1 2 3)
list_length(my_list)
assert("${__ans}" EQUAL 3)
```

And because I want to shield the user from this implementation detail I also provide `ans*` and `return*` functions which allow for a better user experience:

```cmake
## the function
function(list_length list_ref)
    list(LENGTH "${list_ref}" length)
    return(${length})
endfunction()

set(my_list 1 2 3)
list_length(my_list)
ans(the_length)
assert("${the_length}" EQUAL 3)
```

This is of course a bit more verbose but in the long run opens up alot of options known by most programming languages. And also can provide a mechanism to replace bad design choices by the cmake team like `generator-expressions`. But that is another subject not covered in this document.

Having this convention in place allows return values to be passed along without needing to know which of the function parameters are output vars.  This allows a functional style of programming which is less likely to create errors and more likely to be consistent.  And it allows you more freedom:

```cmake
function(list_length list_ref)
    list(LENGTH ${list_ref} length)
    return(${length})
endfunction()

function(list_peek_front list_ref)
    list_length(${list_ref})
    ans(length) ## get result of list_length invocation
    if(NOT length)
        return() ## return nothing
    endif()
    list(GET ${list_ref} 0 return_value)
    return_ref(return_value) ## returns the content of `return_value`
endfunction()

function(list_peek_back list_ref)
    ... left for your imagination ...
endfunction()

set(my_list a b c)
set(operations length peek_front peek_back)

 #message("my_list:")
foreach(operation ${operations})
    eval_cmake("list_${operation}" (my_list))
    ans(result)
    message("  ${operation}: ${result}")
endforeach()
```

Output:
```
my_list:
  length: 3
  peek_front: a
  peek_back: c
```


### Multiple Return Values

A point of critique if you want to follow this convention is that it only allows for one return value.  This is of course true and can be troublesome. However there are good solutions:
```cmake
## the problem: returning multiple values like in the following function is not possible when using return
function(my_function a_ref b_ref c_ref)
    set(${a_ref} 1 PARENT_SCOPE)
    set(${b_ref} 2 PARENT_SCOPE)
    set(${c_ref} 3 PARENT_SCOPE)
endfunction()

my_function(var1 var2 var3)

assert("${var1}" EQUAL 1)
assert("${var2}" EQUAL 2)
assert("${var3}" EQUAL 3)
```


*Alternative 1: Return a list*
```cmake
function(my_function)
    return(1 2 3)
endfunction()

my_function()
ans_extract(var1 var2 var3) # ans extract assign one list element to every variable

assert("${var1}" EQUAL 1)
assert("${var2}" EQUAL 2)
assert("${var3}" EQUAL 3)


```

The problem here is that lists are an ambigous data structure. So this is not the right choice in all causes

*Alternative 2: Return a Map*

```
function(my_function)
    map_new()
    ans(map)
    map_set(${map} a 1)
    map_set(${map} b 2)
    map_set(${map} c 3)
    return(${map})
endfunction()

my_function()
ans(map)
map_get("${map}" a)
ans(var1)
map_get("${map}" b)
ans(var2)
map_get("${map}" c)
ans(var3)


assert("${var1}" EQUAL 1)
assert("${var2}" EQUAL 2)
assert("${var3}" EQUAL 3)
```

In its pure form this is very verbose and will probably keep you from it - however maps are very powerful and easy to use with the correct helper functions.  So this is the way I mostly return multi values.

*Alternative 2b: Pass a Map*

```
function(my_function map)
    map_set(${map} a 1)
    map_set(${map} b 2)
    map_set(${map} c 3)
    return()
endfunction()

map_new()
ans(map)

my_function(${map})

map_get("${map}" a)
ans(var1)
map_get("${map}" b)
ans(var2)
map_get("${map}" c)
ans(var3)

assert("${var1}" EQUAL 1)
assert("${var2}" EQUAL 2)
assert("${var3}" EQUAL 3)
```

This is very similar to the solution specified above and below and fairly self explanatory.

*Alternative 3: Use an `address`*

```
function(my_function addr_a addr_b addr_c)
    address_set(${addr_a} 1)
    address_set(${addr_b} 2)
    address_set(${addr_c} 3)
    return()
endfunction()

mu_function(a b c)
address_get(a)
ans(var1)
address_get(b)
ans(var2)
address_get(c)
ans(var3)


assert("${var1}" EQUAL 1)
assert("${var2}" EQUAL 2)
assert("${var3}" EQUAL 3)
```

This is almost like the normal way of writing cmake functions. I do not like it because I believe functions should have as little side effects as possible but it is fully valid code and might suite your style of programming.


## Exceptions

But what to do if you encounter a problem in your function and want the functions execution to stop?  This is were my interpretation of exceptions come in:

```cmake
function(possibly_failing_function should_fail)
    if(should_faile)
        throw("I Failed.")
    endif()
    return("everything is ok!")
endfunction()


possibly_failing_function(false) ## will not fail
catch((ex) message(FORMAT "Ohoh:  {ex.message}"))
ans(result)
message("result: ${result}") ## will print `result: everthing is ok!`


possibly_failing_function(false) ## will not fail
catch((ex) message(FORMAT "Ohoh:  {ex.message}")) ## will print `Ohoh: I failed.`
ans(result)
message("result: ${result}") ## will print `result:` because no result was returned.
```


You have alot of possiblities here.  For example listening for the global `on_exception` event which allows you to easily log all exceptions. You can also `rethrow` exceptions which causes them to bubble up.  And you can throw exceptions inside `catch` blocks.

This little example only scratched at the surface of the possibilities.  



## Conclusion

So to sum up:  You can use `CMake` as a almost modern programming language as is `JavaScript`, `Python`, and `C++`  Many concepts are easy to emulate and work suprisingly well.  `cmakepp` implements all the functions necessary for you and over the long run makes coding with `CMake` almost bearable.  

Of course there is still more verbosity than in the most modern languages. But all these modifications I propose lead to something great:  A 100 % compatible syntax which can be used inside `CMake` and is very, very expressive. 



