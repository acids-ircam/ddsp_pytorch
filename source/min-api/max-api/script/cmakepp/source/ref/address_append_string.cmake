function(address_append_string ref str)
  set_property(GLOBAL APPEND_STRING PROPERTY "${ref}" "${str}")
endfunction()