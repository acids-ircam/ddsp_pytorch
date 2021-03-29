
function(map_source_file_get)
  map_tryget("${ARGN}" $map_source_file)
  return_ans()
endfunction()
