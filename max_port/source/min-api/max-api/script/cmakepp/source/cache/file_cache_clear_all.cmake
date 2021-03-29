
function(file_cache_clear_all)
  cmakepp_config(temp_dir)
  ans(temp_dir)
  file(REMOVE_RECURSE "${temp_dir}/file_cache")
endfunction()