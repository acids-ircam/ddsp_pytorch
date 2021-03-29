
function(cmake_read path)
  path_qualify(path)
  cmake_deserialize_file("${path}")
  return_ans()
endfunction()