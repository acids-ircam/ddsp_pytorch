## `(<any>)-><bool>`
##
## returns true if the specified value is an encoded list
## meaning that it needs to be decoded before it will be correct
function(is_encoded_list)
  string_codes()
  eval("
    function(is_encoded_list)
      if(\"\${ARGN}\" MATCHES \"[${bracket_open_code}${bracket_close_code}${semicolon_code}]\")
        set(__ans true PARENT_SCOPE)
      else()
        set(__ans false PARENT_SCOPE)
      endif()

    endfunction()
  ")
  is_encoded_list(${ARGN})
  return_ans()
endfunction()


