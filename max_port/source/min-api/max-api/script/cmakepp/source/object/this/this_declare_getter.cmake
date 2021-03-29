
  function(this_declare_getter function_name_ref)
    obj_declare_getter(${this} _res)
    set(${function_name_ref} ${_res} PARENT_SCOPE)
    return()
  endfunction()