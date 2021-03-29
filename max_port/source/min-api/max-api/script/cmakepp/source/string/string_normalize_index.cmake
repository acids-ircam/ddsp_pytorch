## `(<str:<string>> <index:<int>>)-><int>`
##  
## Normalizes the index "index" of a corresponding input string "str".
## Negative indices are transformed into positive values: length - |index|
## Returns -1 if index is out of bounds (index > length of string or length - |index| + 1 < 0)
##
## **Examples**
##  set(input "abcd")
##  string_normalize_index("${input}" 3) # => 3
##  string_normalize_index("${input}" -2) # => 3
##
##
function(string_normalize_index str index)

  set(idx ${index})
  string(LENGTH "${str}" length)
  if(${idx} LESS 0)
    math(EXPR idx "${length} ${idx} + 1")
  endif()
  if(${idx} LESS 0)
    #message(WARNING "index out of range: ${index} (${idx}) length of string '${str}': ${length}")
    return(-1)
  endif()

  if(${idx} GREATER ${length})
    #message(WARNING "index out of range: ${index} (${idx}) length of string '${str}': ${length}")
    return(-1)
  endif()
  return(${idx})
endfunction()