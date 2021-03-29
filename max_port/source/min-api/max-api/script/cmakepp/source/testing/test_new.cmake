
function(test_create file)

  get_filename_component(test_name "${test}" NAME_WE) 
  # setup a directory for the test
  string_normalize("${test_name}")
  ans(test_dir)
  cmakepp_config(temp_dir)
  ans(temp_dir)
  set(test_dir "${temp_dir}/tests/${test_dir}")
  file(REMOVE_RECURSE "${test_dir}")
  get_filename_component(test_dir "${test_dir}" REALPATH)
  
  map_capture_new(test_dir test_name)
  return_ans()  
endfunction()

