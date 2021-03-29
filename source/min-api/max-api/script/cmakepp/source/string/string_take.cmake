## `(<str_name:<string&>> <match:<string>>)-><str_name:<string&>> string>`
##
## Removes "match" from a string reference "str_name" and returns the "match" string.
## Only matches from the beginning of the string reference.
## 
## **Examples**
##  set(input "word")
##  string_take(input "w") # => input equals "ord", match equals "w"
##  set(input "word")
##  string_take(input "ord") # => input is unchanged, no match is returned
##
##
function(string_take str_name match)
  string(FIND "${${str_name}}" "${match}" index)
  #message("trying to take ${match}")
  if(NOT ${index} EQUAL 0)
    return()
  endif()
  #message("took ${match}")
  string(LENGTH "${match}" len)
  string(SUBSTRING "${${str_name}}" ${len} -1 rest )
  set("${str_name}" "${rest}" PARENT_SCOPE)

  return_ref(match)
endfunction()