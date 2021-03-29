

  function(key_value_store_keys)
    this_get(store_dir)
    file(GLOB keys RELATIVE "${store_dir}" "${store_dir}/*")
    return_ref(keys)
  endfunction()