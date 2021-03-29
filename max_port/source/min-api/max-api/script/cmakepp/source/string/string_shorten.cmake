## `(<str:<string>> <max_length:<int>>)-><string>`
##
## Shortens the string "str" to be at most "max_length" characters long.
## Note on "max_length": max_length includes the shortener string (default 3 chars "...").
## Returns the result in "res".
##
## **Examples**
##  set(input "abcde")
##  string_shorten("${input}" 4) # => "a..."
##  string_shorten("${input}" 3) # => "..."
##  string_shorten("${input}" 2) # => ""
##  string_shorten("${input}" 2 ".") # => "a."
##
##
function(string_shorten str max_length)
  set(shortener "${ARGN}")
  if(shortener STREQUAL "")
    set(shortener "...")
  endif()

  string(LENGTH "${str}" str_len)
  if(NOT str_len GREATER "${max_length}")
    return_ref(str)
  endif()
  
  string(LENGTH "${shortener}" shortener_len)
  math(EXPR max_length "${max_length} - ${shortener_len}")
  
  if(${max_length} LESS 0)
    set(res "")
    return_ref(res)
  endif()

  string(SUBSTRING "${str}" 0 ${max_length} res)
  set(res "${res}${shortener}")
  
  return_ref(res)
endfunction()

