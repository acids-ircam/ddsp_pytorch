## returns the implementation of the function (a string containing the source code)
## this only works for functions files and function strings. CMake does not offer
## a possibility to get the implementation of a defined function or macro.
function(function_string_get func)
    is_function_string(is_string "${func}")
    if (is_string)
        return_ref(func)
        return()
    endif ()

    is_function_ref(is_ref "${func}")
    if (is_ref)
        is_address(${func})
        ans(is_ref_ref)

        if (is_ref_ref)
            address_get(${func})
            ans(res)
            return_ref(res)
            return()
        else ()
            set(${func} ${${func}})
        endif ()
    endif ()


    path("${func}")
    ans(fpath)
    is_function_file(is_file "${fpath}")


    if (is_file)
        load_function(file_content "${fpath}")
        function_string_get("${file_content}")
        ans(file_content)
        return_ref(file_content)
        return()
    endif ()


    is_function_cmake(is_cmake_func "${func}")

    if (is_cmake_func)
        ## todo: just return nothing as func is already correctly defined...
        set(source "macro(${func})\n ${func}(\${ARGN})\nendmacro()")
        return_ref(source)
        return()
    endif ()

    if (NOT (is_string OR is_file OR is_cmake_func))
        message(FATAL_ERROR "the following is not a function: '${func}'")
    endif ()
    return()

    lambda_parse("${func}")
    ans(parsed_lambda)

    if (parsed_lambda)
        return_ref(parsed_lambda)
        return()
    endif ()
endfunction()