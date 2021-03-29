## `(<input:<string>> <index:<int>>)-><string>`
##
## Returns the character at the specified position (index). 
## Indexing of strings starts at 0. Indices less than -1 are translated into "length - |index|"
##
## *Examples*
## set(input "example")
## string_char_at("${input}" 3) # => "m"
## string_char_at("${input}"-3) # => "l"
##
##
function(string_char_at input index)
  string(LENGTH "${input}" len)
  string_normalize_index("${input}" ${index})
  ans(index)
  
  if(${index} LESS 0 OR ${index} EQUAL ${len} OR ${index} GREATER ${len}) 
    return()
  endif()
  
  string(SUBSTRING "${input}" ${index} 1 res)
  
  return_ref(res)
endfunction()