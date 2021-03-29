## `(<input:<string>>)-><string>`
##
## Trims the string, by removing whitespaces at the beginning and end.
## 
## **Examples**
##  string_tolower("  whitespaces  ") # => "whitespaces"
##
##
function(string_trim input)
  string(STRIP "${input}" input)
  return_ref(input)
endfunction()