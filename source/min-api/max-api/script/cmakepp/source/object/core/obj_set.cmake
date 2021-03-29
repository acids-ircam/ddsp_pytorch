
  # sets the objects value at ${key}
  function(obj_set this key)

    map_get_special("${this}" "set_${key}")
    ans(setter)
    if(NOT setter)
      map_get_special("${this}" "setter")
      ans(setter)
      if(NOT setter)
        obj_default_setter("${this}" "${key}" "${ARGN}")
        return_ans()
      endif()
    endif()
    set_ans("")
    eval("${setter}(\"\${this}\" \"\${key}\" \"${ARGN}\")")
    return_ans()
  endfunction()