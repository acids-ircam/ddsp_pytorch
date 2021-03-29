

  function(cached retrieve refresh compute_key)
    set(args ${ARGN})
    list_extract_flag(args --refresh)
    ans(refresh_cache)

    if(compute_key STREQUAL "")
      string_combine("_" ${args})
      ans(cache_key)
      string(MD5 "${cache_key}" cache_key)
    else()
      call("${compute_key}"(${args}))
      ans(cache_key)
    endif()
    
    cmakepp_config(temp_dir)
    ans(temp_dir)

    set(cache_dir "${temp_dir}/dir_cache/${cache_key}")
    if(EXISTS "${cache_dir}" NOT refresh_cache)
      call("${retrieve}"(args))
      return_ans()
    endif()

    pushd("${cache_dir}" --create)
    call("${refresh}"(args))
    ans(result)
    popd()

    if(NOT result)
      rm("${cache_dir}")
      return()
    endif()

    call("${retrieve}"(args))
    ans(result)

    return_ref(result)
  endfunction()