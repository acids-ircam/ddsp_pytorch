function(linked_list_push_front linked_list)
  linked_list_insert_before("${linked_list}" "" ${ARGN})
  return_ans()
endfunction()
