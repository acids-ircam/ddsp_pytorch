function(test)

  linked_list_new()
  ans(uut)
  linked_list_push_back(${uut} a)
  ans(a)
  linked_list_push_back(${uut} b)
  ans(b)
  linked_list_push_back(${uut} c)
  ans(c)
  linked_list_push_back(${uut} d)
  ans(d)


  linked_list_remove(${uut} ${c})
  assertf("{b.next}" STREQUAL "${d}")
  assertf("{d.previous}" STREQUAL "${b}")


  linked_list_remove(${uut} ${d})
  assertf("{uut.tail}" STREQUAL "${b}")

  linked_list_remove(${uut} ${a})
  assertf({uut.head} STREQUAL "${b}")

  linked_list_remove(${uut} ${b})
  assertf({uut.head} ISNULL)
  assertf({uut.tail} ISNULL)


endfunction()