function(function_import callable)
    set(args ${ARGN})
    list_extract_flag(args REDEFINE)
    ans(redefine)
    list_extract_flag(args ONCE)
    ans(once)
    list_extract_labelled_value(args as)
    ans(function_name)

    if (callable STREQUAL "")
        message(FATAL_ERROR "no callable specified")
    endif ()

    if (COMMAND "${callable}")
        if ("${function_name}_" STREQUAL "_" OR "${callable}_" STREQUAL "${function_name}_")
            return_ref(callable)
        endif ()
    endif ()


    if (NOT function_name)
        if (COMMAND "${callable}")
            set(function_name "${callable}")
            return_ref(function_name)
        else ()
            function_new()
            ans(function_name)
            set(redefine true)
        endif ()
    endif ()


    if (COMMAND "${function_name}" AND NOT redefine)
        if (once)
            return()
        endif ()
        message(FATAL_ERROR "cannot import '${callable}' as '${function_name}' because it already exists")
    endif ()


    lambda2_tryimport("${callable}" "${function_name}")
    ans(res)
    if (res)
        return_ref(function_name)
    endif ()

    function_string_get("${callable}")
    ans(function_string)

    function_string_rename("${function_string}" "${function_name}")
    ans(function_string)

    function_string_import("${function_string}")

    return_ref(function_name)
endfunction()