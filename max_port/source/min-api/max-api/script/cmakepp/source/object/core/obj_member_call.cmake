# 
function(obj_member_call this key)
  #message("obj_member_call ${this}.${key}(${ARGN})")
  map_get_special("${this}" "member_call")
  ans(member_call)
  if(NOT member_call)
    obj_default_member_call("${this}" "${key}" ${ARGN})
    return_ans()
    #set(member_call obj_default_callmember)
  endif()
  call("${member_call}" ("${this}" "${key}" ${ARGN}))
  return_ans()
endfunction()

