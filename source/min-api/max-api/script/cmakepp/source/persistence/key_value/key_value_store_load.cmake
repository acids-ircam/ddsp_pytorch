
  function(key_value_store_load key)
    this_get(store_dir)
    if(NOT EXISTS "${store_dir}/${key}")
      return()
    endif()
    qm_read("${store_dir}/${key}")
    return_ans()
  endfunction()
