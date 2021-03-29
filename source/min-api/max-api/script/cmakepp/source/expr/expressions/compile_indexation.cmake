
## sets ast.ref, ast.value, ast.code
function(compile_indexation ast)
  map_tryget("${ast}" indexation_lhs)
  ans(lhs)

  map_tryget("${ast}" indexation_elements)
  ans(elements)

  map_tryget("${lhs}" value)
  ans(lhs_value)

  map_tryget("${lhs}" expression_type)
  ans(lhs_expression_type)


  next_id()
  ans(ref)

  set(code "set(${ref})\n")



  set(value_type any)

  list(LENGTH elements element_count)
  if("${element_count}" GREATER 1)
    set(value_type list)
  endif()

  foreach(element ${elements})

    map_tryget("${element}" expression_type)
    ans(expression_type)


    map_tryget("${element}" value)
    ans(element_value)

    if("${expression_type}" STREQUAL "range")
      set(value_type list)
      set(code "${code}
value_range_get(\"${lhs_value}\" \"${element_value}\")
list(APPEND ${ref} \${__ans} )
")
    elseif("${lhs_expression_type}" STREQUAL "ellipsis")
      set(code "${code}foreach(local ${lhs_value})
        get_property(__ans GLOBAL PROPERTY \"\${local}.__object__\" SET)
        if(__ans)
          message(FATAL_ERROR object_get_not_supported_currently)
        else()
          get_property(__ans GLOBAL PROPERTY \"\${local}.${element_value}\")
          list(APPEND ${ref} \${__ans})
        endif()
      endforeach()
        ")
    else()

      set(code "${code}get_property(__ans GLOBAL PROPERTY \"${lhs_value}.__object__\" SET)
                if(__ans)
                  message(FATAL_ERROR object_get_not_supported_currently)
                else()
                  get_property(__ans GLOBAL PROPERTY \"${lhs_value}.${element_value}\")
                  list(APPEND ${ref} \${__ans})
                endif()\n")
    endif()

  endforeach()

#  _message("${code}")
  
  map_set("${ast}" ref "${ref}")
  map_set("${ast}" value "\${${ref}}")
  map_set("${ast}" code "${code}")
endfunction()
