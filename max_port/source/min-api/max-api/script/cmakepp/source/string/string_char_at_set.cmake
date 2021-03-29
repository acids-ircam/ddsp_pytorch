## `(<input:<string>> <index:<int>> <char:<string>>)-><string>`
##
## Sets the character at the specified position (index) to the input 'char'. 
## Indexing of strings starts at 0. Indices less than -1 are translated into "length - |index|"
## 
## **Examples**
##  set(input "example")
##  string_char_at_set("${input}" 0 "E")  # => "Example"
##  string_char_at_set("${input}" 2 "A")  # => "exAmple"
##  string_char_at_set("${input}" -2 "E") # => "examplE"
## 
##
function(string_char_at_set input index char)
  string(LENGTH "${input}" len)
  string_normalize_index("${input}" ${index})
  ans(index)

  if(${index} LESS 0 OR ${index} EQUAL ${len} OR ${index} GREATER ${len}) 
    return()
  endif()

  string(SUBSTRING "${input}" 0 ${index} pre_str)
  MATH(EXPR index "${index} + 1")
  string(SUBSTRING "${input}" ${index} -1 post_str)
  
  set(res "${pre_str}${char}${post_str}")

  return_ref(res)
endfunction()
