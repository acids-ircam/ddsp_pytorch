
# returns the objects value at ${key}
function(obj_get this key)
  map_get_special("${this}" "get_${key}")
  ans(getter)
  if(NOT getter)
    map_get_special("${this}" "getter")
    ans(getter)    
    if(NOT getter)
      obj_default_getter("${this}" "${key}")
      return_ans()
    endif()

  endif()
  set_ans("")
  eval("${getter}(\"\${this}\" \"\${key}\")")
  return_ans()
endfunction()


