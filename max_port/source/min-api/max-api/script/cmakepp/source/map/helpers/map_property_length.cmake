


  ## returns the length of the specified property
  function(map_property_length map prop)
    map_tryget("${map}" "${prop}")
    ans(val)
    list(LENGTH val len)
    return_ref(len)
  endfunction()


  macro(map_property_length map prop)
    get_property(__map_property_length_value GLOBAL PROPERTY "${map}.${prop}")
    list(LENGTH __map_property_length_value __ans)
  endmacro()


  macro(map_property_string_length map prop)
    get_property(__map_property_length_value GLOBAL PROPERTY "${map}.${prop}")
    string(LENGTH "${__map_property_length_value}" __ans)
  endmacro()