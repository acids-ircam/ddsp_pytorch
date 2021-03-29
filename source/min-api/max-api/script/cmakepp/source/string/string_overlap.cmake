## `(<lhs:<string>> <rhs:<string>>)-><string>`
##  
## Returns the overlapping part of input strings "lhs" and "rhs".
## Starts at first char and continues until chars don't match.
##
## **Examples**
##  set(input1 "abcd")
##  set(input2 "abyx")
##  string_overlap("${input1}" "${input2}") # => "ab"
##  set(input2 "wxyz")
##  string_overlap("${input1}" "${input2}") # => ""
##
##
function(string_overlap lhs rhs)
  string(LENGTH "${lhs}" lhs_length)
  string(LENGTH "${rhs}" rhs_length)

  math_min("${lhs_length}" "${rhs_length}")
  ans(len)

  math(EXPR last "${len}-1")

  set(result)

  foreach(i RANGE 0 ${last})
    string_char_at("${lhs}" ${i})
    ans(l)
    string_char_at("${rhs}" ${i})
    ans(r)
    if("${l}" STREQUAL "${r}")
      set(result "${result}${l}")
    else()
      break()
    endif()
  endforeach()

  return_ref(result)
endfunction()
