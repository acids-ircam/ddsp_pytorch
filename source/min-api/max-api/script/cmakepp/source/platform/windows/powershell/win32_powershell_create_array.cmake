
## creates a  powershell array from the specified args
function(win32_powershell_create_array)

    ## compile powershell array for argument list
    set(arg_list)
    foreach(arg ${ARGN})
      string_encode_delimited("${arg}" \")
      ans(arg)
      list(APPEND arg_list "${arg}")
    endforeach()
    string_combine("," ${arg_list})
    ans(arg_list)
    set("${arg_list}" "@(${arg_list})")

    return_ref(arg_list)

endfunction()
