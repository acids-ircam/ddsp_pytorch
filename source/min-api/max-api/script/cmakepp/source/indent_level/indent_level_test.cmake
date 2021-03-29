function(test)



  indent_level_push(0)

  indent("asd" "...")
  ans(res)
  assert(${res} STREQUAL "asd")

  indent_level_push(+1)
  ans(storedlevel)
  indent("asd" "...")
  ans(res)
  assert(${res} STREQUAL "...asd")

  indent_level_push(+1)
  indent_level()
  ans(lvl)
  assert(${lvl} EQUAL 2)
  indent("asd" "...")
  ans(res)
  assert(${res} STREQUAL "......asd")


  indent_level_push()
  indent_level()
  ans(lvl)
  assert(${lvl} EQUAL 3)


  indent_level_restore(${storedlevel})
  indent_level()
  ans(lvl)
  assert(${lvl} EQUAL 1)

  
  

  indent_level_pop()


endfunction()