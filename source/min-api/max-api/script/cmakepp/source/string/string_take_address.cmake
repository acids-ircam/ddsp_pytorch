## `(<str_ref:<string&>>)-><str_ref:<string&>> <string>`
##
## Removes an address (regex format: ":[1-9][0-9]*") from a string reference and returns the address in "res".
## The address is also removed from the input string reference (str_ref).
##
## **Examples**
##
##
function(string_take_address str_ref)
  string_take_regex("${str_ref}" ":[1-9][0-9]*")
  ans(res)
  set(${str_ref} ${${str_ref}} PARENT_SCOPE)   
  return_ref(res)
endfunction()
