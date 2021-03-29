function(set_ans )
  set(__set_ans_val ${ARGN})
  return_ref(__set_ans_val)
endfunction()