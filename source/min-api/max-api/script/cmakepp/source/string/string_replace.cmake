## `(<str:<string>> <pattern:<string>> <replace:<string>>)-><string>`
##
## Replaces all occurences of "pattern" with "replace" in the input string "str".
##
## **Examples**
##  set(input "abca")
##  string_replace("a" "z" "${input}") # => "zbcz"
##  set(input "aaa")
##  string_replace("a" "z" "${input}") # => "zzz"
##
##
function(string_replace str pattern replace)
  string(REPLACE "${pattern}" "${replace}" res "${str}")
  return_ref(res)
endfunction()