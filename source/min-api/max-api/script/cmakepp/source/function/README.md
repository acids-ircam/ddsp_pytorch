## Functions



CMake is a function oriented language. Every line in a cmake script is a function just a function call. It is the only available statement.  CMake does not allow dynamic function calling (ie calling a function which you first know at runtime). This has problem and some further funcitonality issues are addressed in this section.

Functions in cmake are not variables - they have a separate global only scope in which they are defined.  
*A Note on Macros* Macros are also functions.  They do not have their own scope and evaluate arguments differently. They will more likely than not have unintended side effects because of the way the are evaluated. There are valid reasons to use macros but if you do not know them, you SHOULD NOT use macros...




### Datatypes

  * `<cmake code> ::= <string>` any valid cmake code
  * `<cmake function file> ::= <cmake file>` a cmake script file containing a single function 
  * `<function string> :: <cmake code>` a string containing a single function
  * `<cmake file> ::= <path>` a file containing valid cmake code
  * `<function call> ::=<function?!>(<any...>)` a function call can be evaluated to a valid cmake code line which executes the function specified
  * `<function>` ::= <identifier>` any cmake function or macro name for which `if(COMMAND <function>)` evaluates to true.  This can be directly called
  * `<function?!>` ::= <function>|<cmake function file>|<lambda>|<function&>|<function string>|<function string&>  a function?! can be any type of code which somehow evaluates to a function
  * `<function info> ::= {type:<function type>, name:<identifier>, args:<arg ...>, code:<function string>|<function call>}` a map containing information on a specific function. if possible the info map contains the original source code of the function

### Function List


* [anonymous_function](#anonymous_function)
* [anonymous_function_new](#anonymous_function_new)
* [arguments_anonymous_function](#arguments_anonymous_function)
* [arguments_cmake_code](#arguments_cmake_code)
* [arguments_cmake_string](#arguments_cmake_string)
* [arguments_encoded_list](#arguments_encoded_list)
* [arguments_extract](#arguments_extract)
* [arguments_foreach](#arguments_foreach)
* [arguments_function](#arguments_function)
* [arguments_sequence](#arguments_sequence)
* [arguments_string](#arguments_string)
* [bind](#bind)
* [call](#call)
* [check_function](#check_function)
* [curry_compile_encoded_list](#curry_compile_encoded_list)
* [define_function](#define_function)
* [function_capture](#function_capture)
* [function_define_new](#function_define_new)
* [function_help](#function_help)
* [function_import](#function_import)
* [function_import_dispatcher](#function_import_dispatcher)
* [function_import_table](#function_import_table)
* [function_lines_get](#function_lines_get)
* [function_new](#function_new)
* [function_parse](#function_parse)
* [function_signature_get](#function_signature_get)
* [function_signature_regex](#function_signature_regex)
* [function_string_get](#function_string_get)
* [function_string_import](#function_string_import)
* [function_string_rename](#function_string_rename)
* [invocation_argument_encoded_list](#invocation_argument_encoded_list)
* [invocation_argument_string](#invocation_argument_string)
* [invocation_arguments_sequence](#invocation_arguments_sequence)
* [is_anonymous_function](#is_anonymous_function)
* [is_function](#is_function)
* [is_function_cmake](#is_function_cmake)
* [is_function_file](#is_function_file)
* [is_function_ref](#is_function_ref)
* [is_function_string](#is_function_string)
* [load_function](#load_function)
* [rcall](#rcall)
* [save_function](#save_function)
* [try_call](#try_call)
* [wrap_platform_specific_function](#wrap_platform_specific_function)


### Function Descriptions

## <a name="anonymous_function"></a> `anonymous_function`





## <a name="anonymous_function_new"></a> `anonymous_function_new`





## <a name="arguments_anonymous_function"></a> `arguments_anonymous_function`





## <a name="arguments_cmake_code"></a> `arguments_cmake_code`

 `(<start index> <end index>)-> <cmake code>`
 
 captures the cmake code for the calling function and returns it as a string  




## <a name="arguments_cmake_string"></a> `arguments_cmake_string`



 recreates the invocation arguments which when evaluated are identical
 to the arguments passed to the function where this macro was invoked




## <a name="arguments_encoded_list"></a> `arguments_encoded_list`



 returns an encoded list for the specified arguments passed to
 current function invocation.

 you can only use this inside of a cmake function and not inside a macro




## <a name="arguments_extract"></a> `arguments_extract`





## <a name="arguments_foreach"></a> `arguments_foreach`





## <a name="arguments_function"></a> `arguments_function`





## <a name="arguments_sequence"></a> `arguments_sequence`





## <a name="arguments_string"></a> `arguments_string`

 returns the argument string which was passed to the parent function
 it takes into considerations quoted arguments
 todo: start and endindex




## <a name="bind"></a> `bind`





## <a name="call"></a> `call`





## <a name="check_function"></a> `check_function`





## <a name="curry_compile_encoded_list"></a> `curry_compile_encoded_list`

 (["[" <capture vars> "]"] <callable> "(" (<argument>|<assignment>)* ")" ["=>" <?func_name>(<arg names>)  ])->






## <a name="define_function"></a> `define_function`

 define an inline function
 e.g. `define_function(my_func() message(hello))`




## <a name="function_capture"></a> `function_capture`

 captures variables from the current scope in the function




## <a name="function_define_new"></a> `function_define_new`

 `(<signarture> <cmake code>)-><function name>`

 defines an anonymous cmake function and returns its reference (name)
 the code needs to begin with the signature without the name e.g. `(arg1 arg2)` 
  




## <a name="function_help"></a> `function_help`





## <a name="function_import"></a> `function_import`





## <a name="function_import_dispatcher"></a> `function_import_dispatcher`





## <a name="function_import_table"></a> `function_import_table`





## <a name="function_lines_get"></a> `function_lines_get`





## <a name="function_new"></a> `function_new`





## <a name="function_parse"></a> `function_parse`





## <a name="function_signature_get"></a> `function_signature_get`





## <a name="function_signature_regex"></a> `function_signature_regex`





## <a name="function_string_get"></a> `function_string_get`





## <a name="function_string_import"></a> `function_string_import`





## <a name="function_string_rename"></a> `function_string_rename`





## <a name="invocation_argument_encoded_list"></a> `invocation_argument_encoded_list`





## <a name="invocation_argument_string"></a> `invocation_argument_string`





## <a name="invocation_arguments_sequence"></a> `invocation_arguments_sequence`





## <a name="is_anonymous_function"></a> `is_anonymous_function`





## <a name="is_function"></a> `is_function`





## <a name="is_function_cmake"></a> `is_function_cmake`





## <a name="is_function_file"></a> `is_function_file`





## <a name="is_function_ref"></a> `is_function_ref`





## <a name="is_function_string"></a> `is_function_string`





## <a name="load_function"></a> `load_function`





## <a name="rcall"></a> `rcall`





## <a name="save_function"></a> `save_function`





## <a name="try_call"></a> `try_call`





## <a name="wrap_platform_specific_function"></a> `wrap_platform_specific_function`

 defines the function called ${function_name} to call an operating system specific function
 uses ${CMAKE_SYSTEM_NAME} to look for a function called ${function_name}${CMAKE_SYSTEM_NAME}
 if it exists it is wrapped itno ${function_name}
 else ${function_name} is defined to throw an error if it is called





