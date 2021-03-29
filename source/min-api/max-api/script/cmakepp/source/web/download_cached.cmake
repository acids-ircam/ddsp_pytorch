
  ## downloadsa the specified url and stores it in target file
  ## if specified
  ## --refresh causes the cache to be updated
  ## --readonly allows optimization if the result is not modified
  function(download_cached uri)
    set(args ${ARGN})
    list_extract_flag(args --refresh)
    ans(refresh)
    list_extract_flag(args --readonly)
    ans(readonly)
    
    cmakepp_config(cache_dir)
    ans(cache_dir)

    string(MD5 cache_key "${uri}")
    set(cached_path "${cache_dir}/download_cache/${cache_key}")
   
    if(EXISTS "${cached_path}" AND NOT refresh)
      if(readonly)
        glob("${cached_path}/**")
        ans(file_path)
        if(EXISTS "${file_path}")
          return_ref(file_path)
        endif()
        rm("${cached_path}")
      else()
        message(FATAL_ERROR "not supported")
      endif()
    endif()

    mkdir("${cached_path}")
    download("${uri}" "${cached_path}" ${args})
    ans(res)
    if(NOT res)
      rm("${cached_path}")
    endif()
    return_ref(res)
  endfunction()
