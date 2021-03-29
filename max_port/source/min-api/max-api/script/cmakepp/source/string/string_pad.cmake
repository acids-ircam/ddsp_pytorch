## `(<str:<string>> <len:<int>> <argn:<string>>)-><string>`
##  
## Pads the specified string to be as long as specified length "len".
##  - If the string is longer then nothing is padded
##  - If no delimiter is specified than " " (space) is used
##  - If "--prepend" is specified for "argn" the padding is inserted at the beginning of "str"
##
## **Examples**
##  set(input "word")
##  string_pad("${input}" 6) # => "word  "
##  string_pad("${input}" 4) # => "word"
##
##
function(string_pad str len)  
  set(delimiter ${ARGN})
  list_extract_flag(delimiter --prepend)
  ans(prepend)
  if("${delimiter}_" STREQUAL "_")
    set(delimiter " ")
  endif()  
  string(LENGTH "${str}" actual)  
  if(${actual} LESS ${len})
    math(EXPR n "${len} - ${actual}") 

    string_repeat("${delimiter}" ${n})
    ans(padding)
    
    if(prepend)
      set(str "${padding}${str}")
    else()
      set(str "${str}${padding}")    
    endif()    
  endif()
  return_ref(str)
endfunction()