## `(<str:<string>>)-><bool>`
##  
## Returns true if the input string "str" is empty 
## Note: cmake evals "false", "no" which 
##       destroys tests for real emtpiness
##
## **Examples**
##  set(input "")
##  string_isempty("${input}") # => true
##  set(input "false")
##  string_isempty("${input}") # => false
##
##
 function(string_isempty  str)    
    if("_" STREQUAL "_${str}")
      return(true)
    endif()
    return(false)
 endfunction()