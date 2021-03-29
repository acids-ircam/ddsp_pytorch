
  function(key_value_store_delete key)
    this_get(store_dir)
    if(EXISTS "${store_dir}/${key}")
      rm("${store_dir}/${key}")
      return(true)
    endif()
    return(false)
  endfunction()