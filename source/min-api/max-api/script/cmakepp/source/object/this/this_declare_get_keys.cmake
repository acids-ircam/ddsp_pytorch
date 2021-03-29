
  function(this_declare_get_keys function_ref)
    obj_declare_get_keys(${this} _ref)
    set(${function_ref} ${_ref} PARENT_SCOPE)
  endfunction()