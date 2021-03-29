## `(<input:<string>>)-><string>`
##  
## Replaces all non-alphanumerical characters in the string "input" with an underscore 
##
## **Examples**
##  set(input "a?")
##  string_normalize("${input}") # => "a_"
##  set(input "a bc .")
##  string_normalize("${input}") # => "a bc _"
##
##
function(string_normalize input)
	string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" res "${input}")
	return_ref(res)
endfunction()