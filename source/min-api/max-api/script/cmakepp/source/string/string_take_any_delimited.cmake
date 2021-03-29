## `(<str_ref:<string&>> <delimiters:<delimiter:<string>>...>>)-><str_ref:<string&>> <string>`
##
## Removes delimiters of a string and the undelimited string is returned.
## The undelimited string is also removed from the input string reference (__str_ref).
## Notes on the delimiter:
##  - Can be a list of delimiters
##  - Beginning and end delimiter can be specified
##  - May only be a single char
##  - Escaped delimiters are unescaped
##
## **Examples**
##  set(in_ref_str "'a string'")
##  string_take_any_delimited(in_ref_str ') # => in_ref_str equals "" and match equals "a string"
##  set(in_ref_str "\"a string\", <another one>")
##  string_take_any_delimited(in_ref_str "'', <>") # => in_ref_str equals "\"a string\"" and match equals "another one"
##
##
function(string_take_any_delimited str_ref)
  foreach(delimiter ${ARGN})
    string(LENGTH "${${str_ref}}" l1)
    string_take_delimited(${str_ref} "${delimiter}")
    ans(match)
    string(LENGTH "${${str_ref}}" l2)

    if(NOT "${l1}" EQUAL "${l2}")
      set("${str_ref}" "${${str_ref}}" PARENT_SCOPE)
      return_ref(match)
    endif()
  endforeach()

  return()
endfunction()
