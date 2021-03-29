## `(<lhs:<string&>> <rhs:<string&>>)-><lhs:<string&>> <rhs:<string&>>`
##
## Removes the beginning of the string that matches
## from reference string "lhs" and "rhs". 
## See **Examples** for passing references.
##
## **Examples**
##  set(in_lhs "simple test")
##  set(in_rhs "simple a")
##  string_trim_to_difference(in_lhs in_rhs) # => in_lhs equals "test", in_rhs equals "a" 
##  set(in_lhs "a test")
##  set(in_rhs "b test")
##  string_trim_to_difference(in_lhs in_rhs) # => in_lhs equals "a test", in_rhs equals "b test" 
##
##
function(string_trim_to_difference lhs rhs)
  string_overlap("${${lhs}}" "${${rhs}}")
  ans(overlap)

  string_take(${lhs} "${overlap}")
  string_take(${rhs} "${overlap}")

  set("${lhs}" "${${lhs}}" PARENT_SCOPE)
  set("${rhs}" "${${rhs}}" PARENT_SCOPE)
endfunction()
