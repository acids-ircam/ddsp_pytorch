##
##
##
function(query_where query)
  data("${query}")
  ans(query)

  map_keys("${query}")
  ans(selectors)

  set(result)

  foreach(selector ${selectors})
    map_tryget(${query} "${selector}")
    ans(predicate)
    if("${selector}" STREQUAL " ")
      set(selector)
      set(foreach_item false)
    elseif("${selector}" MATCHES "(.*)\\[.*\\]")
      set(foreach_item true)
      set(target_property ${CMAKE_MATCH_1})
    else()
      set(foreach_item false)
    endif()

    ref_nav_get("${ARGN}" ${selector})
    ans(value)

    query_literal("${predicate}" __query_predicate)
    ans(success)



    if(success)
      set(matched_values)
      set(found_match false)
      if(foreach_item)
        foreach(item ${value})
          __query_predicate(${item})
          if(__ans)
            list(APPEND matched_values ${item})
            set(found_match true)
          endif()
        endforeach()
      else()
        __query_predicate(${value})
        if(__ans)
          list(APPEND matched_values ${value})
          set(found_match true)
        endif()
      endif()

      if(found_match)
        ref_nav_set("${result}" "!${target_property}" ${matched_values})
        ans(result)
      endif()
    endif()

  endforeach()

  return_ref(result)

endfunction()