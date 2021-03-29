## `(<direcotry> [--algorthm <checksum algorithm> = "MD5"])-><checksum>`
##
## calculates the checksum for the specified directory 
## just like checksum_layout however also factors in the file's contents
## 
function(checksum_dir dir)
  set(args ${ARGN})
  list_extract_labelled_keyvalue(args --algorithm)
  ans(algorithm)

  path_qualify(dir)
  file(GLOB_RECURSE files RELATIVE "${dir}" "${dir}/**")
  if(args)
    list(REMOVE_ITEM files ${ARGN})
  endif()
  checksum_files("${dir}" ${files} ${algorithm})
  return_ans()
endfunction()


