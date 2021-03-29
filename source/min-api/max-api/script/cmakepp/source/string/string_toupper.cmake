## `(<input:<string>>)-><string>`
##
## Transforms the specified string to upper case.
## 
## **Examples**
##  string_tolower("lower") # => "LOWER"
##
##
function(string_toupper input)
  string(TOUPPER "${input}" input)
  return_ref(input)
endfunction()