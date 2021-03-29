
## returns all identifiers for specified file data directory
function(file_data_ids dir)
  path("${dir}")
  ans(dir)
  glob("${dir}/*.cmake")

  ans(files)
  set(keys)
  foreach(file ${files})
    path_component("${file}" --file-name)
    ans(key)
    list(APPEND keys "${key}")
  endforeach()
  return_ref(keys)
endfunction()