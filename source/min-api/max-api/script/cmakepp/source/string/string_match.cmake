## `(<input:<string>>)-><bool>`
##  
## Evaluates string "str" against regex "regex".
## Returns true if it matches.
##
## **Examples**
##  set(input "a?")
##  string_match("${input}" "[a-z]+\\?") # => true
##  set(input "a bc .")
##  string_match("${input}" "^b") # => false
##
##
function(string_match  str regex)
  if("${str}" MATCHES "${regex}")
    return(true)
  endif()
  return(false)
endfunction()