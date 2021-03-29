
  # returns all keys for the specified object
  function(obj_keys obj)
    map_get_special("${obj}" get_keys)
    ans(get_keys)
    if(NOT get_keys)
      obj_default_get_keys("${obj}")
      return_ans()
    endif()
    set_ans("")
    eval("${get_keys}(\"\${obj}\")")
    return_ans()
  endfunction()