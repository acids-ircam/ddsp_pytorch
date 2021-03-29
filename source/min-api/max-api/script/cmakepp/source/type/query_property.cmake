

  ## queries a property
  function(query_property input_callback property)
    property_def("${property}")
    ans(property)
    map_tryget(${property} "display_name")
    ans(display_name)
    map_tryget(${property} "property_type")
    ans(property_type)
    type_def("${property_type}")
    ans(property_type)
    map_tryget(${property_type} type_name)
    ans(property_type_name)
    message("enter ${display_name} (${property_type_name})")
    query_type("${input_callback}" "${property_type}")
    ans(res)
    return_ref(res)
  endfunction()

  

