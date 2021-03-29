## `(<str:<string>> <search:<string>>)-><bool>`
##  
## Returns true if the input string "str" contains "search"
##
## **Examples**
##  set(input "endswith")
##  string_contains("${input}" "with") # => true
##  string_contains("${input}" "swi") # => true
##
##
function(string_contains str search)
  string(FIND "${str}" "${search}" index)
  if("${index}" LESS 0)
    return(false)
  endif()
  return(true)
endfunction()