#uncompresses specific files from archive specified by varargs and stores them in target_dir directory
function(uncompress_file target_dir archive)
  set(files ${ARGN})

  path_qualify(archive)

  mime_type("${archive}")
  ans(types)


  if("${types}" MATCHES "application/x-gzip")
    pushd("${target_dir}" --create)
      tar_lean(-zxvf "${archive}" ${files})
      ans_extract(error)
      ans(result)
    popd()
    return_ref(result)
  else()
    message(FATAL_ERROR "unsupported compression: '${types}'")
  endif()

endfunction()
