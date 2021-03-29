

##  args => contains unparsed arguments
macro(arguments_extract_defined_values __start_arg_index __end_arg_index __name)
  arguments_extract_defined_value_map("${__start_arg_index}" "${__end_arg_index}" "${__name}")
  ans_extract(__map)
  ans(___temp)
  map_import_properties_all("${__map}")
  set(__ans "${___temp}")
  ## ans now contains rest of input list (unparsed args)
  ## 
endmacro()