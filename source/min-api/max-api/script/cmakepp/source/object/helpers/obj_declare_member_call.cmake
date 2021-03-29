
  function(obj_declare_member_call obj function_ref) 
    function_new()
    ans(func)
    map_set_special(${obj} member_call ${func})
    set(${function_ref} ${func} PARENT_SCOPE)
  endfunction()