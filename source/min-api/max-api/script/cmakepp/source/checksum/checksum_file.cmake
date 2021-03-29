## `(<file> [--algorithm <checksum algorithm> = "MD5"])-><checksum>`
##
## calculates the checksum for the specified file delegates the
## call to `CMake`'s file(<algorithm>) function
## 
function(checksum_file file)

  path_qualify(file)

  set(args ${ARGN})
  list_extract_labelled_value(args --algorithm)
  ans(checksum_alg)
  if(NOT checksum_alg)
    set(checksum_alg MD5)
  endif()
  file(${checksum_alg} "${file}" checksum)
  return_ref(checksum)
endfunction()


