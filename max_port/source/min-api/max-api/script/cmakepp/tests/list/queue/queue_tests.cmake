function(test)
    


  queue_new()
  ans(queue)

  queue_isempty(${queue})
  ans(isempty)
  assert( isempty)

  queue_peek(${queue})
  ans(res)
  assert(NOT res)


  queue_push(${queue} "my;long;value")

  queue_peek(${queue})
  ans(res)
  assert(EQUALS ${res} my long value)

  queue_isempty(${queue})
  ans(res)
  assert(NOT res)

  queue_pop(${queue})
  ans(res)

  assert(EQUALS ${res} my long value)
  queue_isempty(${queue})
  ans(res)
  assert(res)


  


endfunction()