## `(<str:<string>> <search:<string>>)-><bool>`
##
## Returns true if "str" starts with the string "search"
## 
## **Examples**
##  string_starts_with("substring" "sub") # => true
##  string_starts_with("substring" "ub") # => false
##
##
function(string_starts_with str search)
  string(FIND "${str}" "${search}" out)
  if("${out}" EQUAL 0)
    return(true)
  endif()
  return(false)
endfunction()