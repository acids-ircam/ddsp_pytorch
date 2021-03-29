function(test)

  process_list()
  ans(res)
  list(LENGTH res len)
  assert(NOT ${len} LESS 1)

endfunction()