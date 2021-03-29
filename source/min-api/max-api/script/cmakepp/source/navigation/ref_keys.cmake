


  function(ref_keys ref)
    map_get_special("${ref}" object)
    ans(isobject)
    if(isobject)
      obj_keys("${ref}")
    else()
      map_keys("${ref}")
    endif()
    return_ans()
  endfunction()
