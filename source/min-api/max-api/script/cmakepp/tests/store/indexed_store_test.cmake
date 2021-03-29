function(test)

  indexed_store("store")
  ans(uut)


  assign(res = uut.index_add("id"))
  assign(res = uut.index_add("value"))
  assign(res = uut.index_add("{name:'combined',selector:'[](it)format({it.id}{it.value})'}")) 
  

  timer_start(t1)
  assign(key = uut.save("{id:'abc',value:'123'}"))
  assign(key = uut.save("{id:'abc',value:'1234'}"))
  assign(key = uut.save("{id:'abcd',value:'1234'}"))
  assign(key = uut.save("{id:'abcd',value:'12344'}"))
  timer_print_elapsed(t1)

  timer_start(t1)
  assign(result = uut.find_keys("id==abc"))
  timer_print_elapsed(t1)
  print_vars(result)

  timer_start(t1)
  assign(found = uut.find("combined==abc1234"))
  timer_print_elapsed(t1)

  print_vars(found)
  assertf("{found.id}" STREQUAL "abc")
  assertf("{found.value}" STREQUAL "1234")


  assign(itm = uut.find_keys("value==12344"))

  assign(rest = uut.find_keys("id==abcd"))
  assert(${rest} COUNT 2)

  assign(success = uut.delete("${itm}"))
  
  assert(success)

  assign(rest = uut.find_keys("id==abcd"))
  assert(${rest} COUNT 1)


  assign(itms = uut.keys())
  ans(res)
  assert(${res} COUNT 3)


  assign(itms = uut.list())
  ans(res)
  assert(${res} COUNT 3)
  print_vars(res)

endfunction()