
## `(<linked list> <where: <linked list node> = <linked list>.head)-><linked list node>`
##
## inserts a new linked list node into the linked list before where and returns it.
function(linked_list_insert_before linked_list where)
  linked_list_node_new(${ARGN})
  ans(node)

  if(NOT where)
    map_tryget(${linked_list} tail)
    ans(where)
    if(NOT where)
      map_set(${linked_list} head ${node})
      map_set(${linked_list} tail ${node})
      return(${node})
    endif()
  endif() 

  map_tryget(${where} previous)
  ans(previous)

  map_set_hidden(${node} next ${where})
  map_set_hidden(${node} previous ${previous})
  map_set_hidden(${where} previous ${node})

  if(previous)
    map_set_hidden(${previous} next ${node})
  else()
    map_set(${linked_list} head ${node})
  endif()

  return(${node})
endfunction()
