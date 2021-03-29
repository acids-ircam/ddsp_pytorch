## `()-><linked list>`
## 
## creates a new linked list 
## 
## ```
## <linked list node> ::= <null> | {
##   head: <linked list node>|<null>
##   tail: <linekd list node>|<null>
## }
## ```
function(linked_list_new)
  map_new()
  ans(linked_list) 

  map_set(${linked_list} head)
  map_set(${linked_list} tail)

  return_ref(linked_list)  
endfunction()
