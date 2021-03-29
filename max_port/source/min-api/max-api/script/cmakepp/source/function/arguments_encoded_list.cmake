##
##
## returns an encoded list for the specified arguments passed to
## current function invocation.
##
## you can only use this inside of a cmake function and not inside a macro
macro(arguments_encoded_list __arg_begin __arg_end)
    set(__ans)
    if(${__arg_end} GREATER ${__arg_begin})
        math(EXPR __last_arg_index "${__arg_end} - 1")
        foreach(i RANGE ${__arg_begin} ${__last_arg_index})

            string(REPLACE "[" "" __tmp_encoded_list "${ARGV${i}}")
            string(REPLACE "]" "" __tmp_encoded_list "${__tmp_encoded_list}")
            string(REPLACE ";" "" __tmp_encoded_list "${__tmp_encoded_list}")
            if("${__tmp_encoded_list}_" STREQUAL "_")
                set(__tmp_encoded_list "")
            endif()
            list(APPEND __ans "${__tmp_encoded_list}")
        endforeach()
    endif()
endmacro()