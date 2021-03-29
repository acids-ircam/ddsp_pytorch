##
##
##
##
macro(arguments_extract_typed_values __start_arg_index __end_arg_index)
  arguments_encoded_list("${__start_arg_index}" "${__end_arg_index}")
  ans(__arg_res)
  list_extract_typed_values(__arg_res ${ARGN})
endmacro()
