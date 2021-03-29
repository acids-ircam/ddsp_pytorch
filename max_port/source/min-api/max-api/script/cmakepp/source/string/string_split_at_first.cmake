## `(<parta:<string&>> <partb:<string&>> <input:<string>> <separator:<string>>)-><parta:<string&>> <partb:<string&>>`
##
## Splits the string "input" at the first occurence of "separator" and returns 
## both parts in the string references "parta" and "partb".
## See **Examples** for passing references.
##
## **Examples**
## 
##  set(input "a@@b@@c")
##  string_split_at_first(partA partB "${input}" "@@") # => partA equals "a", partB equals "b@@c"
##
##
function(string_split_at_first parta partb input separator)
  string(FIND "${input}" "${separator}" idx )
  
  if(${idx} LESS 0 OR "${separator}_" STREQUAL "_")
    set(${parta} "${input}" PARENT_SCOPE)
    set(${partb} "" PARENT_SCOPE)
    return()
  endif()

  string(SUBSTRING "${input}" 0 ${idx} pa)
  math(EXPR idx "${idx} + 1")

  string(SUBSTRING "${input}" ${idx} -1 pb)
  set(${parta} ${pa} PARENT_SCOPE)
  set(${partb} ${pb} PARENT_SCOPE)
endfunction()