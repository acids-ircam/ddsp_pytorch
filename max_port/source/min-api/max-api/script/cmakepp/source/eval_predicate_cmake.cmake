
function(eval_predicate_cmake code)
  address_new()
  ans(__temp_address)
  eval("if(${code})\naddress_set(${__temp_address} true)\nelse()\naddress_set(${__temp_address} false)\nendif()")
  #message("if(${code})\naddress_set(${__temp_address} true)\nelse()\naddress_set(${__temp_address} false)\nendif()")
  address_get(${__temp_address})
  return_ans()
endfunction()