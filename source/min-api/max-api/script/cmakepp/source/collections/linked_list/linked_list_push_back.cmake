
  function(linked_list_push_back linked_list)
    linked_list_insert_after("${linked_list}" "" ${ARGN})
    return_ans()
  endfunction()
  