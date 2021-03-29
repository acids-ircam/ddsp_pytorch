function(test)



  # call
  set(callable_string "[](a b)return('{{a}}{{b}}')")
  script(" $callable_string(1,2)")
  ans(res)
  assert("${res}" STREQUAL "12")


script("{valaa1:['asd']}")
ans(res)
assert(DEREF "{res.valaa1}" STREQUAL "asd")

script("{ include:'hello', exclude:'asd'}")
ans(res)
assert(DEREF "{res.include}" STREQUAL "hello")
assert(DEREF "{res.exclude}" STREQUAL "asd")
  
  function(TestClassX)
    message("instance x")
    this_set(x 3)
    this_set(y ${ARGV0})
    this_set(z ${ARGV1})
  endfunction()
  script("new TestClassX('asd','bsd')")
  ans(res)
  assert(DEREF "{res.x}" STREQUAL 3)
  assert(DEREF "{res.y}" STREQUAL asd)
  assert(DEREF "{res.z}" STREQUAL bsd)


  message("${lang}:'${package_dir}/resources/expr.json'")
  
  function(testfu)
    message("argn ${ARGN}")
  endfunction()

  script("  $testfu(()->'muhkuh', ()->'nuhkuh')")
  

  script("(asd_asd, bsd_bsd)->{[bsd_bsd,asd_asd]; }")
  ans(res)
  call("${res}"(a 1))
  ans(res)
  assert(EQUALS ${res} 1 a)

#return()
  # statements
  script("$test1 = 'asd'; $test2='bsd'")

  ans(res)
  assert("${res}" STREQUAL "bsd")
  assert(test1)
  assert("${test1}" STREQUAL "asd")
  assert(test2)
  assert("${test2}" STREQUAL "bsd")
  
  # null coalescing
  set(someMap)
  script("$someMap = $someMap ?? {a:123}")
  ans(res1)
  script("$someMap = $someMap ?? {a:234}")
  ans(res2)
  is_map(${res1} )
  ans(ismap)
  assert(ismap)
  assert("${res1}" STREQUAL "${res2}")
  assert(DEREF "{res2.a}" STREQUAL "123")

  # parenthesesed assignment
  script("$res = ($b = {}).a = 'ad'")
  assert("${res}" STREQUAL "ad")
  assert(DEREF "{b.a}" STREQUAL "ad")

  # assign bound value
  map_new()
    ans(someMap)
  script("$someMap.value1 = 123")
  ans(res)
  assert(${res} STREQUAL "123")
  assert(DEREF "{someMap.value1}" STREQUAL 123) 
  


  # assign indexer
  map_new()
    ans(this)
  script("['testvalue'] = 33")
  ans(res)
  assert("${res}" STREQUAL "33")
  assert(DEREF "{this.testvalue}" STREQUAL 33)

  # chain multiple assign
  map_new()
    ans(this)
  script("asd = bsd = csd = 3")
  ans(res)
  assert("${res}" STREQUAL 3)
  assert(DEREF "{this.asd}" STREQUAL "3")
  assert(DEREF "{this.bsd}" STREQUAL "3")
  assert(DEREF "{this.csd}" STREQUAL "3")
  # assignment of cmake var
  set(ast)
  script("$asd='ad'")
  ans(res)
  assert("${res}" STREQUAL "ad")
  assert(asd)
  assert("${asd}" STREQUAL "ad")

  # assignment of scope variable
  map_new()
  ans(this)
  script("bsd = 'hula'")
  ans(res)
  assert(${res} STREQUAL "hula")
  assert(DEREF "{this.bsd}" STREQUAL "hula")


  # complicated sample
  script(" {a:{b:{c:'[]()return({{this}})',d:'hello'}}}.a.b.c().d")
  ans(res)
  assert("${res}" STREQUAL "hello")

  # object
  script("{}")
  ans(res)
  is_map(${res} )
  ans(ismap)
  assert(ismap)


  # object with value
  script("{asd:312}")
  ans(res)
  is_map(${res} )
  ans(ismap)
  assert(ismap)
  assert(DEREF "{res.asd}" STREQUAL "312")

  #object with multiple values
  script("{asd:'asd', bsd:'bsd', csd:{a:1,b:2}}")
  
  ans(res)
  assert(DEREF "{res.asd}" STREQUAL "asd")
  assert(DEREF "{res.bsd}" STREQUAL "bsd")
  assert(DEREF "{res.csd.a}" STREQUAL "1")
  assert(DEREF "{res.csd.b}" STREQUAL "2")

  # list
  script("[1,2,'abc']")
  ans(res)
  assert(EQUALS ${res} 1 2 "abc")

  # string
  script("'312'")
  ans(res)
  assert("${res}" STREQUAL "312")

  # number
  script("41414")
  ans(res)
  assert("${res}" EQUAL 41414)

  # cmake identifier
  set(cmake_var abcd)
  script("$cmake_var")
  ans(res)
  assert("${res}" STREQUAL "abcd")

  # scope identifier
  map_new()
  ans(this)
  map_set(${this} identifier "1234")
  script("identifier")
  ans(res)
  assert("${res}" STREQUAL "1234")

  # bind 
  map_new()
  ans(this)
  map_new()
  ans(next)
  map_set(${this} a ${next})
  map_set(${next} b "9876")
  script("a.b")
  ans(res)
  assert("${res}" STREQUAL "9876")


  # indexation
  map_new()
  ans(a)
  map_set(${a} a 1234)
  script("$a['a']")
  ans(res)
  assert("${res}" STREQUAL "1234")




endfunction()