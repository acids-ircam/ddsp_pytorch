## `(<base dir> <file...>)-><checksum>`
##
## create a checksum from specified files relative to <dir>
## the checksum is influenced by the files relative paths 
## and the file content 
## 
function(checksum_files dir)
  set(args ${ARGN})
  list_extract_labelled_keyvalue(args --algorithm)
  ans(algorithm)

  list(LENGTH args len)
  if(len)
    list(REMOVE_DUPLICATES args)
    list(SORT args)
  endif()
  
  set(checksums)
  foreach(file ${ARGN})
    if(EXISTS "${dir}/${file}")
      if(NOT IS_DIRECTORY "${dir}/${file}")
        checksum_file("${dir}/${file}" ${algorithm})
        ans(file_checksum)
        # create checksum from file checsum and file name
        checksum_string("${file_checksum}.${file}" ${algorithm})
        ans(combined_checksum)
        list(APPEND checksums "${combined_checksum}")
      endif()
    endif()
  endforeach()

  checksum_string("${checksums}" ${checksum_alg})
  ans(checksum_dir)
  return_ref(checksum_dir)
endfunction()