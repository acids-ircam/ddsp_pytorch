## `(<__str_ref:<string&>>)-><__str_ref:<string&>>`
##
## Removes preceeding whitespaces of the input string reference.
## See **Examples** for passing references.
##
## **Examples**
##  set(in_ref_str "   test")
##  string_take_whitespace(in_ref_str) # => in_ref_str equals "test"
##  set(in_ref_str "   test  ")
##  string_take_whitespace(in_ref_str) # => in_ref_str equals "test  "
##
##
function(string_take_whitespace __string_take_whitespace_string_ref)
  string_take_regex("${__string_take_whitespace_string_ref}" "[ ]+")
  ans(__string_take_whitespace_res)
  set("${__string_take_whitespace_string_ref}" "${${__string_take_whitespace_string_ref}}" PARENT_SCOPE)
  return_ref(__string_take_whitespace_res)
endfunction()

## Faster version
macro(string_take_whitespace __str_ref)
  if("${${__str_ref}}" MATCHES "^([ ]+)(.*)")
    set(__ans "${CMAKE_MATCH_1}")
    set(${__str_ref} "${CMAKE_MATCH_2}")
  else()
    set(__ans)
  endif()
endmacro()


