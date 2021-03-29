function(expr_compile_cmake_identifier)
  #message("cmake_identifier")
  #address_print(${ast})
  map_get(${ast}  children)
  ans(identifier)
  map_get(${identifier}  data)
  ans(identifier)
  
  set(res "
  #expr_compile_cmake_identifier
  if(COMMAND \"${identifier}\")
    set_ans(\"${identifier}\")
  else() 
    set_ans_ref(\"${identifier}\") 
  endif()
  # end of expr_compile_cmake_identifier")
  return_ref(res)
endfunction()