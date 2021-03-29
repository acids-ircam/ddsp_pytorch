

  function(content_dir_update dir)
    set(file_name "${ARGN}")
    if(NOT file_name)
      set(file_name "cache-key.cmakepp")
    endif()

    path("${dir}/${file_name}")
    ans(cache_key_path)

    pushd("${dir}")
    checksum_glob_ignore(** "!${file_name}" --recurse)
    ans(actual_checksum)
    popd()

    fwrite("${cache_key_path}" "${actual_checksum}")



    return_ref(actual_checksum)
  endfunction()
