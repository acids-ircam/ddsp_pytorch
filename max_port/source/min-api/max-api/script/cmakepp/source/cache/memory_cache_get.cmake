

  function(memory_cache_get cache_key)
    set(args ${ARGN})
    list_extract_flag(args --const)
    ans(isConst)

    memory_cache_key("${cache_key}")
    ans(key)
    map_tryget(memory_cache_entries "${key}")
    ans(value)
    if(NOT isConst)
      map_clone_deep("${value}")
      ans(value)
    endif()
#    
    return_ref(value)
  endfunction()


  