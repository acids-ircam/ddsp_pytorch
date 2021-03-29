function(test)
  
  linked_list_new()
  ans(uut)

  linked_list_insert_after(${uut} "" hello)
  ans(node)
  address_get("${node}")
  ans(value)
  assert("${value}" STREQUAL "hello")
  assertf("{uut.head}" STREQUAL "${node}")
  assertf("{uut.tail}" STREQUAL "${node}")



  linked_list_insert_after(${uut} "" hello2)
  ans(node2)
  assertf("{uut.head}" STREQUAL "${node}")
  assertf("{uut.tail}" STREQUAL "${node2}") 


  assertf("{uut.head.next}" STREQUAL "${node2}")
  assertf("{uut.tail.previous}" STREQUAL "${node}")


  linked_list_insert_after(${uut} "${node}" hello3)
  ans(node3)
  assertf("{uut.head}" STREQUAL "${node}")
  assertf("{uut.tail}" STREQUAL "${node2}")
  assertf("{node.previous}" ISNULL)
  assertf("{node2.next}" ISNULL)
  assertf("{node3.previous}" STREQUAL "${node}")
  assertf("{node3.next}" STREQUAL "${node2}")
  assertf("{node.next}" STREQUAL "${node3}")
  assertf("{node2.previous}" STREQUAL "${node3}")
endfunction()