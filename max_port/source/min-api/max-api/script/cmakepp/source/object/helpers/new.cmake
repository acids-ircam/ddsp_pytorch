# shorthand for map_new and obj_new
# accepts a Type (which has to be a cmake function)
function(new)
  obj_new(${ARGN})
  return_ans()
endfunction()


  