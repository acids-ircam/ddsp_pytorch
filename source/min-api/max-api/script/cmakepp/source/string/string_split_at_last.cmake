## `(<parta:<string&>> <partb:<string&>> <input:<string>> <separator:<string>>)-><parta:<string&>> <partb:<string&>>`
##
## Splits the string "input" at the last occurence of "separator" and returns 
## both parts in the string references "parta" and "partb".
## See **Examples** for passing references.
##
## **Examples**
##  set(input "a@@b@@c")
##  string_split_at_last(partA partB "${input}" "@@") # => partA equals "a@@b", partB equals "c"
##
##
function(string_split_at_last parta partb input separator)
  string(FIND "${input}" "${separator}" idx  REVERSE)

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