
function(map_filter_template_key map scope)
  map_keys("${map}")
  ans(keys)

  foreach(key ${keys})
    eval_predicate_template_cmake("${scope}" "${key}")
    ans(result)

    if(result)
      return_ref(key)
    endif()
  endforeach()  
  return()
endfunction()