
function(obj_injectable_callmember this key)
  map_get_special("${this}" before_call)
  ans(before_call)
  map_get_special("${this}" after_call)
  ans(after_call)

  set(call_this ${this})
  set(call_args ${ARGN})
  set(call_key ${key})
  set(call_result)
  
  if(before_call)
    call("${before_call}"())
  endif()
  obj_default_member_call("${this}" "${key}" "${ARGN}")
  ans(call_result)
  if(after_call)
    call("${after_call}"())
  endif()
  return_ref(call_result)
endfunction()


function(obj_before_callmember obj func)
  map_set_special("${obj}" call_member obj_injectable_callmember)
  map_set_special("${obj}" before_call "${func}")
endfunction()

function(obj_after_callmember obj func)
  map_set_special("${obj}" call_member obj_injectable_callmember)
  map_set_special("${obj}" after_call "${func}")
endfunction()