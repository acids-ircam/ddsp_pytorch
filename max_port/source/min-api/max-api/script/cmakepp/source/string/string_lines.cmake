## `(<input:<string>>)-><string...>`
##  
## Splits the specified string "input" into lines
## Caveat: The string would have to be semicolon encoded
##         to correctly display lines with semicolons 
##
## **Examples**
##  set(input "a\nb")
##  string_lines("${input}") # => "a;b"
##  set(input "a b\nc")
##  string_lines("${input}") # => "a b;c"
##
##
function(string_lines input)      
  string_split("${input}" "\n")

  return_ans(lines)
endfunction()
