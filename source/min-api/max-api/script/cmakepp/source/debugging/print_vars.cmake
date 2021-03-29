## prints the specified variables names and their values in a single line
## e.g.
## set(varA 1)
## set(varB abc)
## print_vars(varA varB)
## output:
##  varA: '1' varB: 'abc'
function(print_vars)
  set(__print_vars_args "${ARGN}")
  list_extract_flag(__print_vars_args --plain)
  ans(__print_vars_plain)
  set(__str)
  foreach(__print_vars_arg ${__print_vars_args})
    assign(____cur = ${__print_vars_arg})
    if(NOT __print_vars_plain)
      json("${____cur}")
      ans(____cur)
    else()
      set(____cur "'${____cur}'")
    endif()

    string_shorten("${____cur}" "300")
    ans(____cur)
    set(__str "${__str} ${__print_vars_arg}: ${____cur}")

  endforeach()
  message("${__str}")
endfunction()