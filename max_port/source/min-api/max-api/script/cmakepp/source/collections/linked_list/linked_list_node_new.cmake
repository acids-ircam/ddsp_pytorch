
## `(<any>...)-><linked list node>`
## 
## creates a new linked list node which contains the value specified
## 
function(linked_list_node_new)
  map_new()
  ans(node)
  map_set_special(${node} $type linked_list_node)
  address_set(${node} ${ARGN})
  return(${node})
endfunction()