  # calls the object itself
  function(obj_call obj)
    map_get_special("${obj}" "call")
    ans(call)

    if(NOT call)
      message(FATAL_ERROR "cannot call '${obj}' - it has no call function defined")
    endif()
    set(this "${obj}")
    call("${call}" (${ARGN}))
    ans(res)
    return_ref(res )
  endfunction()