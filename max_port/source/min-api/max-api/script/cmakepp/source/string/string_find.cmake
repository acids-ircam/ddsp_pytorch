## `(<str:<string>> <substr:<string>>)-><int>`
##  
## Returns the position where the "substr" was found 
## in the input "str", otherwise -1. 
## NOTE: The flag REVERSE causes the last position of "substr"
##       to be returned
##
## **Examples**
##  set(input "endswith")
##  string_find("${input}" "with") # => 4
##  string_find("${input}" "swi") # => 3
##
##
function(string_find str substr)
  set(args ${ARGN})
  list_extract_labelled_keyvalue(args --reverse REVERSE)
  ans(reverse)
  string(FIND "${str}" "${substr}" idx ${reverse})
  return_ref(idx)
endfunction()