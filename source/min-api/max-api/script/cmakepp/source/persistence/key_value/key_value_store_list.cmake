

  function(key_value_store_list)
    key_value_store_keys()
    ans(keys)
    set(values)
    foreach(key ${keys})
      key_value_store_load("${key}")
      ans_append(values)
    endforeach()  
    return_ref(values)
  endfunction()

  