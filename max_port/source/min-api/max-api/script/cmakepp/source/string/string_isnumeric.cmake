## `(<str:<string>>)-><bool>`
##  
## Returns true if the input string "str" is a positive integer 
## including "0"
##
## **Examples**
##  set(input "1")
##  string_isnumeric("${input}") # => true
##  set(input "-1")
##  string_isnumeric("${input}") # => false
##
##
function(string_isnumeric str)
  if("_${str}" MATCHES "^_[0-9]+$")
    return(true)
  endif()
  return(false)
endfunction()