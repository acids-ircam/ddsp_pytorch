function(test)

  process_list()
  ans(p)
  list_pop_front(p)
  ans(p)


  process_info(${p})
  ans(info)

  assert(DEREF {info.pid} STREQUAL {p.pid})
endfunction()