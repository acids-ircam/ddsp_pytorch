# dynamic function call method
# can call the following
# * a cmake macro or function
# * a cmake file containing a single function
# * a lambda expression (see lambda())
# * a object with __call__ operation defined
# * a property reference ie this.method()
# CANNOT  call 
# * a navigation path
# no output except through return values or referneces
function(call __function_call_func __function_call_paren_open)

    return_reset()
    set(__function_call_args ${ARGN})

    list_pop_back(__function_call_args)
    ans(__function_call_paren_close)

    if (NOT "_${__function_call_paren_open}${__function_call_paren_close}" STREQUAL "_()")
        message("open ${__function_call_paren_open} close ${__function_call_paren_close}")
        message(WARNING "expected opening and closing parentheses for function '${__function_call_func}' '${ARGN}' '${__function_call_args}'")
    endif ()

    if (COMMAND "${__function_call_func}")
        set_ans("")
        eval("${__function_call_func}(\${__function_call_args})")
        return_ans()
    endif ()


    if (DEFINED "${__function_call_func}")
        call("${${__function_call_func}}" (${__function_call_args}))
        return_ans()
    endif ()

    is_address("${__function_call_func}")
    ans(isref)
    if (isref)
        obj_call("${__function_call_func}" ${__function_call_args})
        return_ans()
    endif ()

    propis_address("${__function_call_func}")
    ans(ispropref)
    if (ispropref)
        propref_get_key("${__function_call_func}")
        ans(key)
        propref_get_ref("${__function_call_func}")
        ans(ref)

        obj_member_call("${ref}" "${key}" ${__function_call_func})

    endif ()

    lambda2_tryimport("${__function_call_func}" __function_call_import)
    ans(success)
    if (success)
        __function_call_import(${__function_call_args})
        return_ans()
    endif ()


    if (DEFINED "${__function_call_func}")
        call("${__function_call_func}" (${__function_call_args}))
        return_ans()
    endif ()


    is_function(is_func "${__function_call_func}")
    if (is_func)
        function_import("${__function_call_func}" as __function_call_import REDEFINE)
        __function_call_import(${__function_call_args})
        return_ans()
    endif ()

    if ("${__function_call_func}" MATCHES "^[a-z0-9A-Z_-]+\\.[a-z0-9A-Z_-]+$")
        string_split_at_first(__left __right "${__function_call_func}" ".")
        is_address("${__left}")
        ans(__left_isref)
        if (__left_isref)
            obj_member_call("${__left}" "${__right}" ${__function_call_args})
            return_ans()
        endif ()
        is_address("${${__left}}")
        ans(__left_isrefref)
        if (__left_isrefref)
            obj_member_call("${${__left}}" "${__right}" ${__function_call_args})
            return_ans()
        endif ()
    endif ()

    nav(__function_call_import = "${__function_call_func}")
    if (__function_call_import)
        call("${__function_call_import}" (${__function_call_args}))
        return_ans()
    endif ()

    message(FATAL_ERROR "tried to call a non-function:'${__function_call_func}'")
endfunction()