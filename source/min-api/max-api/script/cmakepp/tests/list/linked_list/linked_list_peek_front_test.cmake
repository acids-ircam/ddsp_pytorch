function(test)

  linked_list_new()
  ans(uut)

  linked_list_push_back("${uut}" 1)
  ans(node1)
  linked_list_push_back("${uut}" 2)

  linked_list_peek_front("${uut}")
  ans(res)
  assert("${res}" STREQUAL "1")

  linked_list_peek_front("${uut}" --node)
  ans(res)
  assert("${res}" STREQUAL "${node1}")




endfunction()