function(test)

  range_instanciate(-1 "1:3:1")
  ans(uut)

  range_partial_unpack(uut)
  assert(${uut.inclusive_begin} EQUALS true)
  assert(${uut.inclusive_end} EQUALS true)
  assert(${uut.begin} EQUALS 1)
  assert(${uut.end} EQUALS 3)
  assert(${uut.length} EQUALS 3)
  assert(${uut.increment} EQUALS 1)
endfunction()