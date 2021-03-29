


  ## queries a type
  function(query_type input_callback type)
    type_def("${type}")
    ans(type)

    map_tryget(${type} properties)
    ans(properties)

    list(LENGTH properties is_complex)

    if(NOT is_complex)
      query_fundamental("${input_callback}" "${type}")
      ans(res)
    else()
      query_properties("${input_callback}" "${type}")
      ans(res)      
    endif()
    return_ref(res)
  endfunction()  