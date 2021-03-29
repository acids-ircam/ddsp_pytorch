function(test)



  linked_list_new()
  ans(uut)

  linked_list_insert_before(${uut} "" hello)
  ans(node)
  assert(node)
  address_get("${node}")
  ans(value)
  assert("${value}" STREQUAL "hello")
  assertf("{uut.head}" STREQUAL "${node}")
  assertf("{uut.tail}" STREQUAL "${node}")


  linked_list_insert_before(${uut} "" hello2)
  ans(node2)
  assertf("{uut.head}" STREQUAL "${node2}" )
  assertf("{uut.tail}" STREQUAL "${node}" )
  assertf("{node.previous}" STREQUAL "${node2}")
  assertf("{node2.next}" STREQUAL "${node}")

  linked_list_insert_before(${uut} "${node}" "hello3")
  ans(node3)
  assertf("{uut.head}" STREQUAL "${node2}")
  assertf("{uut.tail}" STREQUAL "${node}")
  assertf("{node2.next}" STREQUAL "${node3}")
  assertf("{node.previous}" STREQUAL "${node3}")
  assertf("{node.next}" ISNULL)
  assertf("{node2.previous}" ISNULL)

endfunction()