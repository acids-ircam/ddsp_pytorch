
  function(obj_declare_getter obj function_name_ref)
      function_new()
      ans(func)
      map_set_special(${obj} getter "${func}")
      set(${function_name_ref} ${func} PARENT_SCOPE)
      return()
  endfunction()
