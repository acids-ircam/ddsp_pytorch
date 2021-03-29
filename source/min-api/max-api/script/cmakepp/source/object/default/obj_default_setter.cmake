


  # default setter for object properties sets the
  # owned value @ key
  function(obj_default_setter obj key value)
    map_set("${obj}" "${key}" "${value}")
    return()
  endfunction()