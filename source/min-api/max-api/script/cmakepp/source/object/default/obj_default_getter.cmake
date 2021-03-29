  # default getter for object properties tries to get
  # the maps own value and if not looks for the prototype
  # special field and calls obj_get on it
  function(obj_default_getter obj key)
    map_has("${obj}" "${key}")
    ans(has_own_property)
    if(has_own_property)
      map_tryget("${obj}" "${key}")
      return_ans()  
    endif()

    map_get_special("${obj}" "prototype")
    ans(prototype)
    #message("proto is ${prototype}")
    if(NOT prototype)
      return()
    endif()

    obj_get("${prototype}" "${key}")
    return_ans()
  endfunction()