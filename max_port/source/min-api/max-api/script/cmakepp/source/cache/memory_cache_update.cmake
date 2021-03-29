

  function(memory_cache_update cache_key value)
    set(args ${ARGN})
    list_extract_flag(args --const)
    ans(isConst)
    if(NOT isConst)
        map_clone_deep("${value}")
        ans(value)
    endif()

    memory_cache_key("${cache_key}")
    ans(key)
    
    map_set_hidden(memory_cache_entries "${key}" "${value}")
  endfunction()