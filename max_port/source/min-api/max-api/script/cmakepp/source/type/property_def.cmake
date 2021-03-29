
    ## parses a property 
  function(property_def prop)
    data("${prop}")
    ans(prop)
    is_map("${prop}")
    ans(ismap)

    if(ismap)
      return_ref(prop)
    endif()

    map_new()
    ans(res)


    string_take_regex(prop "[^:]+")
    ans(prop_name)

    if("${prop}_" STREQUAL "_")
      set(prop_type "any")
    else()
      string_take(prop :)
      set(prop_type "${prop}")
    endif()


    map_set(${res} property_name "${prop_name}")
    map_set(${res} display_name "${prop_name}")
    map_set(${res} property_type "${prop_type}")
    return_ref(res)
  endfunction()