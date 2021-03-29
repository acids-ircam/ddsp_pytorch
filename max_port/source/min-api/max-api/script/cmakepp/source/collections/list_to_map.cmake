
  function(list_to_map lst key_selector)
    function_import("${key_selector}" as __to_map_key_selector REDEFINE)
    map_new()
    ans(res)
    foreach(item ${${lst}})
      __to_map_key_selector(${item})
      ans(key)
      map_set(${res} "${key}" "${item}")
    endforeach()
    return_ref(res)

  endfunction()
