
function(test)
  fwrite_data("package1.json" "{asd:'asd'}" --json)

  timer_start(t1)
  fopen_data("package1")
  ans(res)
  timer_print_elapsed(t1)

  assertf({res.asd} STREQUAL "asd")


  fwrite_data("package2.scmake" "{asd:'bsd'}" )

  timer_start(t2)
  fopen_data(package2)
  ans(res)
  timer_print_elapsed(t2)

  assertf({res.asd} STREQUAL "bsd")

  fwrite_data("package3.qm" "{asd:'csd'}" )

  timer_start(t3)
  fopen_data(package3)
  ans(res)
  timer_print_elapsed(t3)

  assertf({res.asd} STREQUAL "csd")


endfunction()
