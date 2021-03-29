
function(obj_declare_call obj out_function_name)
  function_new()
  ans(callfunc)
  map_set_special("${obj}" call "${callfunc}")
  set("${out_function_name}" "${callfunc}" PARENT_SCOPE)  
endfunction()  