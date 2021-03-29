
  function(content_dir_check dir )
    set(file_name ${ARGN})
    if(NOT file_name)
      set(file_name "cache-key.cmakepp")
    endif()

    path("${dir}/${file_name}")
    ans(cache_key_path)

    if(NOT EXISTS "${cache_key_path}")
      return(false)
    endif()

    fread("${cache_key_path}")
    ans(expected_checksum)
    pushd("${dir}")
    checksum_glob_ignore(** "!${file_name}" --recurse)
    ans(actual_checksum)
    popd()

    if(NOT "${expected_checksum}_" STREQUAL "${actual_checksum}_")
      return(false)
    endif()


    return(true)
  endfunction()
    