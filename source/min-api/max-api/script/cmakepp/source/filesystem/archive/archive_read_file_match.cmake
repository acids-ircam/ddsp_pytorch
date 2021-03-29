
function(archive_read_file_match archive regex)
  path_qualify(archive)
  archive_match_files("${archive}" "${regex}")
  ans(file_path)
  list(LENGTH file_path count)
  if(NOT "${count}" EQUAL 1)
    return()
  endif()

  archive_read_file("${archive}" "${file_path}")
  return_ans()
endfunction()

