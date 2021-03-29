## `(<str:<string>> <search:<string>>)-><bool>`
##  
## Returns true if the input string "str" ends with "search"
##
## **Examples**
##  set(input "endswith")
##  string_ends_with("${input}" "with") # => true
##  string_ends_with("${input}" "width") # => false
##
##
function(string_ends_with str search)
  string(FIND "${str}" "${search}" out REVERSE)
  if(${out} EQUAL -1)
    return(false)
  endif()
  string(LENGTH "${str}" len)
  string(LENGTH "${search}" len2)
  math(EXPR out "${out}+${len2}")
  if("${out}" EQUAL "${len}")
    return(true)
  endif()
  return(false)
endfunction()