##
## returns the cmake function that this lambda was compiled to
function(lambda2 source)
  lambda2_instanciate("${source}")
  ans(lambda)
  map_tryget(${lambda} function_name)
  return_ans()
endfunction()

