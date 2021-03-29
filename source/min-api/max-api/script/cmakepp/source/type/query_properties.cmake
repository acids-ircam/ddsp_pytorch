
  function(query_properties input_callback type)

    map_new()
    ans(res)

    message_indent_push()
    foreach(property ${properties})
      property_def("${property}")
      ans(property)
      query_property("${input_callback}" "${property}")
      ans(value)
      map_tryget(${property} property_name)
      ans(prop_name)
      map_set(${res} "${prop_name}" "${value}")
    endforeach()
    message_indent_pop()
    return_ref(res)
  endfunction()
