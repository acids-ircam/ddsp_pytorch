
  function(this_declare_member_call function_ref)
    obj_declare_member_call(${this} _res)
    set(${function_ref} ${_res} PARENT_SCOPE)
  endfunction()
