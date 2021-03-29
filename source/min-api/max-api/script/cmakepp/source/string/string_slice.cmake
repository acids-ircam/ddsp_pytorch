## `(<str:<string>> <start_index:<int>> <end_index:<int>>)-><string>`
##
## Extracts a portion from string "str" at the specified index: [start_index, end_index)
## Indexing of slices starts at 0. Indices less than -1 are translated into "length - |index|"
## Returns the result in "result".
##
## **Examples**
##  set(input "abc")
##  string_slice("${input}" 0 1) # => "a"
##  set(input "abc")
##  string_slice("${input}" 0 2) # => "ab"
##
##
function(string_slice str start_index end_index)
  # indices equal => select nothing

  string_normalize_index("${str}" ${start_index})
  ans(start_index)
  string_normalize_index("${str}" ${end_index})
  ans(end_index)
  
  if(${start_index} LESS 0)
    message(FATAL_ERROR "string_slice: invalid start_index ")
  endif()
  if(${end_index} LESS 0)
    message(FATAL_ERROR "string_slice: invalid end_index")
  endif()
  # copy array
  set(result)
  math(EXPR len "${end_index} - ${start_index}")
  string(SUBSTRING "${str}" ${start_index} ${len} result)

  return_ref(result)
endfunction()
  