## sets the a setter functions for a specific property
  function(obj_declare_property_setter obj property_name setter)
    set(args ${ARGN})
    list_extract_flag(args --hidden)
    ans(hidden)
    function_new()
    ans("${setter}")
    if(NOT hidden)
      map_set("${obj}" "${property_name}" "")
    endif()
    map_set_special("${obj}" "set_${property_name}" "${${setter}}")
    set("${setter}" "${${setter}}" PARENT_SCOPE)

  endfunction()
