## `(<str:<string>>)-><string>`
##  
## Escapes chars used by regex strings in the input string "str".
## Escaped characters: "\ / ] [ * . - ^ $ ? ) ( |"
##
## **Examples**
##  set(input "()")
##  string_regex_escape("${input}") # => "\(\)"
##  set(input "no_escape")
##  string_regex_escape("${input}") # => "no_escape"
##
##
function(string_regex_escape str)
  #string(REGEX REPLACE "(\\/|\\]|\\.|\\[|\\*)" "\\\\\\1" str "${str}")
  string(REGEX REPLACE "(\\/|\\]|\\.|\\[|\\*|\\$|\\^|\\-|\\+|\\?|\\)|\\(|\\|)" "\\\\\\1" str "${str}")
  return_ref(str)
endfunction()
