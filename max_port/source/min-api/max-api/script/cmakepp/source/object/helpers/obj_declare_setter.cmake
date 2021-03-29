
  function(obj_declare_setter obj function_ref)
    function_new()
    ans(res)
    map_set_special(${obj} setter ${res})
    set(${function_ref} ${res} PARENT_SCOPE)
  endfunction()
