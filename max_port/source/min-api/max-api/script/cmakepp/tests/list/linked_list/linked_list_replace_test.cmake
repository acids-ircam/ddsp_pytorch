function(test)


  linked_list_new()
  ans(uut)



  linked_list_insert_after("${uut}" "" a)
  ans(a)
  linked_list_insert_after(${uut} "" b)
  ans(b)
  linked_list_insert_after(${uut} "" c)
  ans(c)

  linked_list_replace("${uut}" "${a}")
  ans(d)
  assertf("{uut.head}" STREQUAL "${d}")
  assertf("{uut.tail}" STREQUAL "${c}")
  assertf("{b.previous}" STREQUAL "${d}")
  assertf("{d.next}" STREQUAL "${b}")

  linked_list_replace("${uut}" "${b}")
  ans(e)
  assertf("{uut.head}" STREQUAL "${d}")
  assertf("{uut.tail}" STREQUAL "${c}")
  assertf("{e.previous}" STREQUAL "${d}")
  assertf("{d.next}" STREQUAL "${e}")
  assertf("{e.next}" STREQUAL "${c}")
  assertf("{c.previous}" STREQUAL "${e}")


  linked_list_replace("${uut}" "${c}")
  ans(f)
  assertf("{uut.head}" STREQUAL "${d}")
  assertf("{uut.tail}" STREQUAL "${f}")
  assertf("{f.previous}" STREQUAL "${e}")
  assertf("{e.next}" STREQUAL "${f}")
  








endfunction()