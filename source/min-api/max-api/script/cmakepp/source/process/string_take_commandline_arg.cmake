
  function(string_take_commandline_arg str_ref)
    string_take_whitespace(${str_ref})
    set(regex "(\"([^\"\\\\]|\\\\.)*\")|[^ ]+")
    string_take_regex(${str_ref} "${regex}")
    ans(res)
    if(NOT "${res}_" STREQUAL _)
      set("${str_ref}" "${${str_ref}}" PARENT_SCOPE)
    endif()
    if("${res}" MATCHES "\".*\"")
      string_take_delimited(res "\"")
      ans(res)
    endif()

    return_ref(res)


  endfunction()