## `(<__str_ref:<string&>>)-><__str_ref:<string&>> <string>`
##
## Removes delimiters of a string and the undelimited string is returned.
## The undelimited string is also removed from the input string reference (__str_ref).
## Notes on the delimiter:
##  - Default is double quote ""
##  - Beginning and end delimiter can be specified
##  - May only be a single char
##  - Escaped delimiters are unescaped
##
## **Examples**
##  set(in_ref_str "'a string'")
##  string_take_delimited(in_ref_str ') # => in_ref_str equals "" and res equals "a string"
##  set(in_ref_str "'a string'")
##  string_take_delimited(in_ref_str "''") # => same as above
##
##
function(string_take_delimited __string_take_delimited_string_ref )
  regex_delimited_string(${ARGN})
  ans(__string_take_delimited_regex)
  string_take_regex(${__string_take_delimited_string_ref} "${__string_take_delimited_regex}")
  ans(__string_take_delimited_match)
  if(NOT __string_take_delimited_match)
    return()
  endif()
  set("${__string_take_delimited_string_ref}" "${${__string_take_delimited_string_ref}}" PARENT_SCOPE)

  # removes the delimiters
  string_slice("${__string_take_delimited_match}" 1 -2)
  ans(res)
  # unescape string
  string(REPLACE "\\${delimiter_end}" "${delimiter_end}" res "${res}")
  return_ref(res) 
endfunction()

## faster version
function(string_take_delimited __str_ref )
  set(input "${${__str_ref}}")

  regex_delimited_string(${ARGN})
  ans(regex)
  if("${input}" MATCHES "^${regex}")
    string(LENGTH "${CMAKE_MATCH_0}" len)
    if(len)
      string(SUBSTRING "${input}" ${len} -1 input )
    endif()
    string(REPLACE "\\${delimiter_end}" "${delimiter_end}" res "${CMAKE_MATCH_1}")
    set("${__str_ref}" "${input}" PARENT_SCOPE)
    set(__ans "${res}" PARENT_SCOPE)
  else()
    set(__ans PARENT_SCOPE)
  endif()

endfunction()

