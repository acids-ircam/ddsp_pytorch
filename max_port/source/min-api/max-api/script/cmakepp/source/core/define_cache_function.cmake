## `()->`
## 
## defines a function called alias which caches its results
##
function(define_cache_function generate_value)
  set(args ${ARGN})

  list_extract_labelled_value(args =>)
  ans(alias)
  if(NOT alias)
    function_new()
    ans(alias)
  endif()

  list_extract_labelled_value(args --generate-key)
  ans(generate_key)
  if(NOT generate_key)
      set(generate_key "[]()checksum_string('{{ARGN}}')")
  endif()

  list_extract_labelled_value(args --select-value)
  ans(select_value)
  if(NOT select_value)
      set(select_value "[]()set_ans('{{ARGN}}')")
  endif()
  

  list_extract_labelled_value(args --cache-dir)
  ans(cache_dir)
  if(NOT cache_dir)
    cmakepp_config(cache_dir)
    ans(cache_dir)
    set(cache_dir "${cache_dir}/cache_functions/${alias}")
  endif()


  list_extract_flag(args --refresh)
  ans(refresh)

#    print_vars(generate_key generate_value select_value refresh  cache_dir)
  if(refresh)
    rm(-r "${cache_dir}")
  endif()
    
  callable_function("${generate_key}")
  ans(generate_key)
  callable_function("${generate_value}")
  ans(generate_value)
  callable_function("${select_value}")
  ans(select_value)

  eval("
    function(${alias})
      set(args \${ARGN})
      list_extract_flag(args --update-cache)
      ans(update)

      ${generate_key}(\${args})
      ans(cache_key)
      set(cache_path \"${cache_dir}/\${cache_key}\")
      
      map_has(memory_cache \"\${cache_path}\")
      ans(has_entry)

      if(has_entry AND NOT update)
  #      message(memhit)
        map_tryget(memory_cache \"\${cache_path}\")
        ans(cached_result)
      elseif(EXISTS \"\${cache_path}/value.scmake\" AND NOT update)
   #     message(filehit)
        cmake_read(\"\${cache_path}/value.scmake\")
        ans(cached_result)
        map_set(memory_cache \"\${cache_path}\" \${cached_result})
      else()
       # if(update)
    #      message(update )
       # else()
     #     message(miss )
      #  endif()
        ${generate_value}(\${args})
        ans(cached_result)
        map_set(memory_cache \"\${cache_path}\" \${cached_result})
        cmake_write(\"\${cache_path}/value.scmake\" \${cached_result})
      endif()
      ${select_value}(\${cached_result})
      return_ans()
    endfunction()
    ")
  return_ref(alias)
endfunction()


