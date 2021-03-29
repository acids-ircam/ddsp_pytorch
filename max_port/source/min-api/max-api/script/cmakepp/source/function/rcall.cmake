# allows a single line call with result 
# ie rcall(some_result = obj.getSomeInfo(arg1 arg2))
function(rcall __rcall_result_name equals __callable)
  set_ans("")
  call("${__callable}" ${ARGN})
  ans(res)
  set(${__rcall_result_name} ${res} PARENT_SCOPE)
  return_ref(res)
endfunction()