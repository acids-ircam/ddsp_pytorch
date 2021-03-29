
  function(obj_getprototype obj)
    map_get_special("${obj}" prototype)
    ans(res)
    return_ref(res)
  endfunction()