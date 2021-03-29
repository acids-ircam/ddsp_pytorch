function(test)

  call("[](it)return({{it}})"(3))
  ans(res)
  assert("${res}" STREQUAL "3")

  call("[](it)set(abc {{it}});return({{abc}})"(3))
  ans(res)
  assert("${res}" STREQUAL "3")

  rcall(res2 = "[](it)set(abc {{it}});return({{abc}})"(3))
  assert("${res2}" STREQUAL "3")


return()
  function_import("[]()return(3)" as asdldigger)

  asdldigger()
  ans(res)
  assert("${res}" EQUAL 3)

  timer_start(lambdarun1000)
  foreach(i RANGE 0 1000)
    asdldigger()
  endforeach()
  timer_print_elapsed(lambdarun1000)


  lambda2_compile(asd "[a b c](k b)asdasd(kk)")
  ans(res)

  lambda2_compile_source("hello('asdas;das{{}}d' {{asdasd}} bebibu 'asd{{hello}}asd asd\\'asd') ;byby('bd;sfsdfsdf')")
  ans(res)

  ## takes about 30ms to compile this lambda
  timer_start(lambda100)
  foreach(i RANGE 0 100)
    lambda2("[i](a b)asdasdasd(kk 'asd' 'bsd{{asd}}')")
  endforeach()
  timer_print_elapsed(lambda100)
  
  
  set(a 123)
  set(b 234)
  set(c 345)
  lambda2("[a b](d e) return('{{ARGN}} {{a}} {{b}} {{c}} {{d}} {{e}}')")
  ans(res)

  set(a 456)
  set(c 567)

  call("${res}"(678 789 890))
  ans(res)
  assert(${res} STREQUAL "890 123 234 567 678 789")




endfunction()