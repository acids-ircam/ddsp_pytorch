
  ## universal get function which allows you to get
  ## from an object or map. only allows property names
  ## returns nothing if navigting the object tree fails
  function(get ref_name _equals nav)
    string(REPLACE "." "\;" nav "${nav}")
    set(nav ${nav})
    list_pop_front(nav)
    ans(part)


    set(current "${${part}}")
    map_get_special("${current}" object)
    ans(isobject)

    if(isobject)
      foreach(part ${nav})
        obj_get("${current}" "${part}")
        ans(current)
        if("${current}_" STREQUAL "_")
          break()
        endif()
      endforeach()
    else()
      foreach(part ${nav})
        map_tryget("${current}" "${part}")
        ans(current)
        if("${current}_" STREQUAL "_")
          break()
        endif()
      endforeach()
    endif()
    
    set("${ref_name}" "${current}" PARENT_SCOPE)
  endfunction()