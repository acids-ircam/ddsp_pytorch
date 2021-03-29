function(test)
  event_new(my_event1)

  event_emit(my_event1 a b c)
  ans(res)

  assert(NOT res)


  event_addhandler(my_event1 "[]() return({{ARGN}}) ") 

  event_emit(my_event1 abc)
  ans(res)
  assert(res)
  asserT("${res}" STREQUAL "abc")







  ## test event methods
  event_new()
  ans(event)


  call(event.add("[]()return({{ARGN}} {{ARGN}})"))
  call(event.add("[]()return({{ARGN}} {{ARGN}})"))
  call(event.add("[]()return({{ARGN}} {{ARGN}})"))
  call2(${event} 1 )
  ans(res)
  assert(${res} EQUALS 1 1 1 1 1 1)

  call(event.remove("[]()return({{ARGN}} {{ARGN}})"))
  ans(res)
  assert(res)
  assertf({event.handlers} ISNULL)


  call(event.add("[]()return({{ARGN}})"))
  ans(handler)
  assert(handler)
  assertf("{event.handlers}" STREQUAL "${handler}")


  call(event.remove("${handler}"))
  ans(res)
  assert(res)


  ## event was already removed.  result is false
  call(event.remove("${handler}"))
  ans(res)
  assert(NOT res)


  call(event.add("[]()return({{ARGN}} {{ARGN}})"))
  ans(handler)


  call2("${handler}" 3 4 5)
  ans(res)
  assert(${res} EQUALS 3 4 5 3 4 5)


  call(event.clear())
  assertf({event.handlers} ISNULL)

return()


  event_new(asdl_event)
  ans(event_1)
  assert(event_1)
  assertf({event_1.event_id} STREQUAL "asdl_event")
  is_callable("${event_1}")
  ans(is_callable)
  assert(is_callable)

  assert(COMMAND asdl_event)

  asdl_event(a b c)
  ans(res)
  assert(NOT res)



  event_addhandler(asdl_event "[]()return({{ARGN}})")
  event_addhandler(asdl_event "[]()return({{ARGN}})")
  event_addhandler(asdl_event "[]()return({{ARGN}})")


  asdl_event(a b c)
  ans(res)
  assert("${res}" EQUALS a b c a b c a b c)



  call2("${event_1}"  asdf)
  ans(res)
  assert(${res} EQUALS asdf)


  event_new()
  ans(res)
  assert(res)
  assertf(NOT "{res.event_id}_" STREQUAL "_")
  is_callable("${res}")
  ans(is_callable)
  assert(is_callable)

  map_tryget("${res}" event_id)
  ans(event)
  assert(COMMAND "${event}")


  ## emitting an event takes about 70 ms
  timer_start(events)
  foreach(i RANGE 0 100)
    asdl_event(${i})
  endforeach()
  timer_print_elapsed(events)

endfunction()