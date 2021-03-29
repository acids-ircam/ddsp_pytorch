
  ## `(<clause:{<selector>:<literal...>}> <any..> )-><bool>`
  ##
  ## queries the specified args for the specified clause
  function(query_disjunction clause)
    map_keys("${clause}")
    ans(selectors)

    foreach(selector ${selectors})
      map_tryget(${clause} "${selector}")
      ans(predicates)

      foreach(predicate ${predicates})

        if("${selector}" STREQUAL " ")
          set(selector)
          set(foreach_item false)
        elseif("${selector}" MATCHES "(.*)\\[.*\\]$")
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
          if(foreach_item)
            foreach(item ${value})
              __query_predicate(${item})
              if(__ans)
                return(true)
              endif()
            endforeach()
          else()
            __query_predicate(${value})
            if(__ans)
              return(true)
            endif()
          endif()
        endif()
      endforeach()
    endforeach()

    return(false)
  endfunction()