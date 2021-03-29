



  function(fallback_data_read dirs id)    
    set(maps )
    foreach(dir ${dirs})
      file_data_read("${dir}" "${id}")
      ans(res)
      list(APPEND maps "${res}")
    endforeach()
    list(REVERSE maps)
    map_merge(${maps})
    ans(res)
    return_ref(res)
  endfunction()
