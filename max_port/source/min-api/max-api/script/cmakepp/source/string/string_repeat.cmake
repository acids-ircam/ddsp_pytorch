## `(<what:<string>> <n:<int>>)-><string>`
##
## Repeats string "what" "n" times and separates them with an optional separator
##
## **Examples**
##  set(input "a")
##  string_repeat("${input}" 2) # => "aa"
##  string_repeat("${input}" 2 "@@") # => "a@@a"
##
##  
function(string_repeat what n)
  set(separator "${ARGN}")
  
  if(${n} LESS 1)
    return()
  endif()

  set(res "${what}")

  if(${n} GREATER 1)
    foreach(i RANGE 2 ${n})
      set(res "${res}${separator}${what}")
    endforeach()
  endif()

  return_ref(res)
endfunction()