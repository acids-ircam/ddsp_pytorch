## `(<string_subject:<string>> <split_regex:<string>>)-><string...>`
##
## Splits the string "input" at the occurence of the regex "split_regex".
## Returns the result in "res".
## TODO: does not handle strings containing list separators properly
##
## **Examples**
##  set(input "a@@b@@c")
##  string_split("${input}" "@@") # => "a;b;c"
##
##
function(string_split  string_subject split_regex)
	string(REGEX REPLACE ${split_regex} ";" res "${string_subject}")
  return_ref(res)
endfunction()
