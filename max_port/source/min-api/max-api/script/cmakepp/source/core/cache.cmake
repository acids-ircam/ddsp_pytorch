function(cached arg)
    json("${arg}")
    ans(ser)
    string(MD5 cache_key "${ser}")
    set(args ${ARGN})
    list(LENGTH args arg_len)
    if(arg_len)

      map_set(global_cache_entries "${cache_key}" "${args}")
      return_ref(args)
    endif()


    map_tryget(global_cache_entries "${cache_key}")    
    ans(res)
    return_ref(res)


endfunction()

  macro(return_hit arg_name)
    cached("${${arg_name}}")
    if(__ans)
      message("hit")
      return_ans()
    endif()
      message("not hit")
  endmacro()


