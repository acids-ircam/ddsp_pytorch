
function(address_push_front ref)
    get_property(value GLOBAL PROPERTY "${ref}")
  set_property( GLOBAL PROPERTY "${ref}" "${ARGN}" "${value}")
endfunction()