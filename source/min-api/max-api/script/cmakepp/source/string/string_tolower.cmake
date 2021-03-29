## `(<input:<string>>)-><string>`
##
## Transforms the specified string to lower case.
## 
## **Examples**
##  string_tolower("UPPER") # => "upper"
##
##
function(string_tolower input)
  string(TOLOWER "${input}" input)
  return_ref(input)
endfunction()