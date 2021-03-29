## `(<str:<string>> <start:<int>> <end:<int>>)-><string>`
##
## Wrapper function for substring.
## Returns a substring of input "str" with the index parameter "start" and optionally "len".
## Note on indexing: len is the amount of chars to be extracted starting from index "start"
## 
## **Examples**
##  string_substring("substring" 1)     # => "ubstring"
##  string_substring("substring" 1 2)   # => "ub"
##  string_substring("substring" -3 2)  # => "ng"
##
##
function(string_substring str start)
  set(len ${ARGN})
  if(NOT len)
    set(len -1)
  endif() 
  string_normalize_index("${str}" "${start}")
  ans(start)

  string(SUBSTRING "${str}" "${start}" "${len}" res)
  return_ref(res)
endfunction()