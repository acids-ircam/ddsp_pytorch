


## `(<linked list> <where: <linked list node> = <linked list>.tail >  <any>... )-><linked list node>`
## 
## inserts a new linked list node after `where`. if where is null then the tail of the list is used.
## the arguments passed after where are used as the value of the new node
function(linked_list_insert_after linked_list where)
  
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

  map_tryget(${where} next)
  ans(next)

  map_set_hidden(${node} previous ${where})
  map_set_hidden(${node} next ${next})
  map_set_hidden(${where} next ${node})
  if(next)
    map_set_hidden(${next} previous ${node})
  else()
    map_set(${linked_list} tail ${node})
  endif()

  return(${node})
endfunction()
