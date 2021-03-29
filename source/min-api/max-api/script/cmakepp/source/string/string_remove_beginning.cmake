## `(<original:<string>> <beginning:<string>>)-><string>`
##
## Removes the beginning "n"-chars of the string "original".
## Number of chars "n" is calculated based on string "beginning".
##
## **Examples**
##  set(input "abc")
##  string_remove_ending("${input}" "a") # => "ab"
##  string_remove_ending("${input}" "ab") # => "a"
##
##
function(string_remove_beginning original beginning)
  string(LENGTH "${beginning}" len)
  string(LENGTH "${original}" orig_len)
  if(len GREATER orig_len)
    set(len 0)
  endif()
  string(SUBSTRING "${original}" ${len} -1 original)

  return_ref(original)
endfunction()