
# default implementation for calling a member
# imports all vars int context scope
# and binds this to the calling object
function(obj_default_member_call this key)
  #message("obj_default_callmember ${this}.${key}(${ARGN})")
  obj_get("${this}" "${key}")
  ans(member_function)
  if(NOT member_function)
    message(FATAL_ERROR "member does not exists '${this}.${key}'")
  endif()
  # this elevates all values of obj into the execution scope
  #obj_import("${this}")  
  call("${member_function}"(${ARGN}))
  return_ans()
endfunction()

