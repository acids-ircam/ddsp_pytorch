# evaluates a truth expression 'if' and returns true or false 
function(eval_truth)
  if(${ARGN})
    return(true)
  endif()
  return(false)
endfunction()