
  function(ref_prop_set ref prop)
    map_get_special("${ref}" object)
    ans(isobject)
    if(isobject)
      obj_set("${ref}" "${prop}" ${ARGN})
    else()
      map_set("${ref}" "${prop}" ${ARGN})
    endif()
  endfunction()

