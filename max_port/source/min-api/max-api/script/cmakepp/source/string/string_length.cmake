## `(<str:<string>>)-><int>`
##  
## Returns the length of the input string "str"
##
## **Examples**
##  set(input "a")
##  string_length("${input}") # => 1
##  set(input "ab c")
##  string_length("${input}") # => 4
##
##
function(string_length str)
  string(LENGTH "${str}" len)
  return_ref(len)
endfunction()