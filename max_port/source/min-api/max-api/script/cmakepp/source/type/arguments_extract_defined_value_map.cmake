

macro(arguments_extract_defined_value_map __start_arg_index __end_arg_index __name)
  arguments_encoded_list("${__start_arg_index}" "${__end_arg_index}")
  set(__current_function_name "${__current_function_name}::${__name}")
  ans(__arg_res)
  parameter_definition_get("${__name}")
  ans(___defs)
  
  if(___defs)
    list_extract_defined_values(__arg_res "${___defs}")
  endif()


  #ans_extract(values)
  #ans(rest)  
endmacro()