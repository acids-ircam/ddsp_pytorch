
  function(linked_list_remove linked_list where)
    map_import_properties("${where}" previous next)

    if(next)
      map_set_hidden("${next}" previous "${previous}")
    else()
      map_set("${linked_list}" tail "${previous}")
    endif()

    if(previous)
      map_set_hidden("${previous}" next "${next}")
    else()
      map_set("${linked_list}" head "${next}")
    endif()

    return()
  endfunction() 