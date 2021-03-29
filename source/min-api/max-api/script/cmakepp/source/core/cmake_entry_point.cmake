## returns the entry script file from which cmake was started
function(cmake_entry_point)
  commandline_args_get()
  ans(args)
  list_extract_labelled_value(args -P)
  ans(script_file)
  path_qualify(script_file)
  return_ref(script_file)
endfunction()