
  ## obj_declare_property_getter(<objref> <propname:string> <getter:cmake function ref>)
  ## declares a property getter for a specific property
  ## after the call getter will contain a function name which needs to be implemented
  ## the getter function signature is (current_object key values...)
  ## the getter function also has access to `this` variable
  function(obj_declare_property_getter obj property_name getter)
    set(args ${ARGN})
    list_extract_flag(args --hidden)
    ans(hidden)
    function_new()
    ans("${getter}")
    if(NOT hidden)
      map_set("${obj}" "${property_name}" "")
    endif()
    map_set_special("${obj}" "get_${property_name}" "${${getter}}")
    set("${getter}" "${${getter}}" PARENT_SCOPE)
  endfunction()