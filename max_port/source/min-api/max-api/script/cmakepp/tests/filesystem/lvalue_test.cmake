function(test)






  # ref property? indexer
  function(lvalue)
    list(LENGTH ARGN len)
    if(${len} EQUAL 4)
      set(__ans ${ARGN} PARENT_SCOPE)
    endif()
    
    string(REGEX REPLACE "([a-zA-Z0-9_\\-]+).*" "\\1" ref "${ARGN}")
    string(REGEX REPLACE "[^\\[]*\\[(.*)\\]" "\\1" indexer "${ARGN}")
    if("${indexer}" STREQUAL "${ARGN}")
      set(indexer 0 *)
    else()
      string(REPLACE - ";" indexer "${indexer}")
      if("${indexer}_" STREQUAL "_")
        set(indexer -1 -1)
      else()
        list(GET indexer 0 low)
        list(REMOVE_AT indexer 0)

        if("${indexer}_" STREQUAL _)
          math(EXPR indexer "${low} + 1")
        elseif(NOT "${indexer}_" STREQUAL "*_")
          matH(EXPR indexer "${indexer} + 1")
        endif()          
        set(indexer "${low}" "${indexer}")
      endif()
    endif()
    string(REGEX REPLACE "[^\\.]+\\.([^\\[])+.*" "\\1" prop "${ARGN}")
    if("${prop}" STREQUAL "${ARGN}")
      set(prop "")
    endif()
    set(res "${ref};${indexer};${prop}")
    return_ref(res)
  endfunction()







  macro(lvalue_unpack __lvalue)
    lvalue(${${__lvalue}})
    ans(${__lvalue})
    list(GET ${__lvalue} 0 ${__lvalue}.ref)
    list(GET ${__lvalue} 1 2 ${__lvalue}.indexer)
    list(GET ${__lvalue} 3 ${__lvalue}.prop)
    if("${${__lvalue}.prop}_" STREQUAL "_" )
      set(${__lvalue}.has_prop false)
    else()
      set(${__lvalue}.has_prop true)
    endif()
    is_map("${${${__lvalue}.ref}}")
    ans(${__lvalue}.is_map)
  endmacro()

  function(lvalue_get lvalue)
    lvalue_unpack(lvalue)
    message("hasprop ${lvalue.has_prop} ismap ${lvalue.is_map}")
    if(lvalue.is_map AND lvalue.has_prop)
      map_tryget(${${lvalue.ref}} ${lvalue.prop})
      ans(val)

    else()
      set(val ${${lvalue.ref}})
    endif()


    list_slice(val ${lvalue.indexer})
    ans(result)

    return_ref(result)




  endfunction()




  function(lvalue_set lvalue)
    lvalue_unpack(lvalue)
    if(lvalue.has_prop)
      if(NOT lvalue.is_map)
        ## cause an error here if path should not be created
        map_new()
        ans(${lvalue.ref})
      endif()
      map_tryget("${${lvalue.ref}}" "${lvalue.prop}")
      ans(val)
      list_replace_slice(val ${lvalue.indexer} ${ARGN})
      map_set("${${lvalue.ref}}" "${lvalue.prop}" ${val})
    else()
      list_replace_slice(${lvalue.ref} ${lvalue.indexer} ${ARGN})
    endif()
    set(${lvalue.ref} ${${lvalue.ref}} PARENT_SCOPE)
  endfunction()





  return()

  set(uut "prop")
  arg_unpack(uut)
  message("is_range '${uut.is_range}', is_property '${uut.is_property}', range '${uut.range}', '${uut.range.begin}:${uut.range.increment}:${uut.range.end}', property '${uut.property}'")

  return()
  set(asd "abc")
  arg_unpack(asd)
  assert(${asd.range.increment} EQUALS 1)
  assert(${asd.range.begin} EQUALS 0)
  assert(${asd.range.end} EQUALS *)

  assert(${asd.property} EQUALS abc)
  assert(${asd.is_range} EQUALS false)
  assert(${asd.is_property} EQUALS true)

  set(asd "[]")
  arg_unpack(asd)
  assert(${arg.range} EQUALS *:*)
  assert(${arg.property} ISNULL)
  assert(${arg.is_range} EQUALS true)
  assert(${arg.is_property} EQUALS false) 

  set(asd "[3]")
  arg_unpack(asd)
  assert(${arg.range} EQUALS 3 4)
  assert(${arg.property} ISNULL)
  assert(${arg.is_range} EQUALS true)
  assert(${arg.is_property} EQUALS false)


  set(asd [3-4])
  arg_unpack(asd)
  assert(${arg.range} EQUALS 3 5)
  assert(${arg.property} ISNULL)

  set(asd "[3-*]")
  arg_unpack(asd)
  assert(${arg.range} EQUALS 3 *)
  assert(${arg.property} ISNULL)

  return()

  function(path_create)
    map_new()
    ans(current_map)
    


    foreach(arg ${ARGN})
      arg_unpack("${arg}")

    endforeach()
    

    set(${first})

  endfunction()






  return()

  function(lvalue_path first)

    map_new()
    ans(root_map)
    map_set(${root_map} ${first} ${${first}})


    set(current_map ${root_map})
    set(current_prop ${first})
    set(current_indexer 0 *)

    set(args ${ARGN})
    list_pop_back(args)
    ans(value)

    foreach(arg ${args})
      set(indexer)
      set(property)
      set(val)
      if("${arg}" MATCHES "^\\[.*\\]$")
        string(REGEX REPLACE "\\[(.*)\\]" "\\1" indexer "${arg}")
      else()
        set(property "${arg}")
      endif()




    endforeach()

    map_set("${current_map}" "${current_prop}" "${value}")


    map_tryget(${root_map} ${first})
    ans(res)
    set(${first} ${res} PARENT_SCOPE)
    return()
  endfunction()


  set(a)
  lvalue_path(a b)
  assert(${a} EQUALS b)

  set(a)
  lvalue_path(a [] b)
  assert(${a} EQUALS b)




return()

  set(a)
  lvalue_set(a.b hi)
  assert(a)
  assertf({a.b} STREQUAL "hi")

  set(a)
  lvalue_set(a.b a b c d)
  assert(a)
  assertf({a.b} EQUALS a b c d)

  obj("{c:2}")
  ans(a)
  lvalue_set(a.b asdasd)
  assertf({a.c} EQUAL 2)
  assertf({a.b} STREQUAL asdasd)

  obj("{b:'asd'}")
  ans(a)
  lvalue_set(a.b kakaka)
  assertf({a.b} STREQUAL kakaka)


  obj("{b:'asd'}")
  ans(a)
  lvalue_set(a.b[] kakaka)
  assertf({a.b} EQUALS asd kakaka)

  obj("{b:[1,2,3,4]}")
  ans(a)
  lvalue_set(a.b[1-2] a b)
  assertf({a.b} EQUALS 1 a b 4)


  obj("{b:[1,2,3,4]}")
  ans(a)
  lvalue_set(a.b[1-*])
  assertf({a.b} EQUALS 1)

  set(a)
  lvalue_set(a hello)
  assert(${a} STREQUAL "hello")

  set(a)
  lvalue_set(a hello hello)
  assert(${a} EQUALS hello hello)

  set(a)
  lvalue_set(a[] hello)
  assert(${a} EQUALS hello)

  set(a byby)
  lvalue_set(a[] hello)
  assert(${a} EQUALS byby hello)


  set(a a b c)
  lvalue_set(a[1] gaga)
  assert(${a} EQUALS a gaga c)

  set(a a b c d)
  lvalue_set(a[1-2] 1 2 3 4)
  assert(${a} EQUALS a 1 2 3 4 d)

  set(a k b c d)
  lvalue_set(a[1-*])
  assert(${a} EQUALS k)



  return()



  set(a 2)
  lvalue_get(a)
  ans(res)
  assert(${res} EQUAL 2)


  set(a 2 3 4)
  lvalue_get(a)
  ans(res)
  assert(${res} EQUALS 2 3 4)


  set(a 2 3 4)
  lvalue_get(a[1])
  ans(res)
  assert(${res} EQUALS 3)


  set(a 2 3 4 5)
  lvalue_get(a[1-2])
  ans(res)
  assert(${res} EQUALS 3 4)



  obj("{b:3}")
  ans(a)
  lvalue_get(a.b)
  ans(res)
  assert(${res} EQUALS 3)


  obj("{b:[3,4,5]}")
  ans(a)
  lvalue_get(a.b[1])
  ans(res)
  assert(${res} EQUALS 4)

  obj("{b:[3,4,5]}")
  ans(a)
  lvalue_get(a.b)
  ans(res)
  assert(${res} EQUALS 3 4 5)





  return()


  lvalue(a)
  ans(res)
  assert(${res} EQUALS a 0 * "")

  lvalue(a[2])
  ans(res) 
  assert(${res} EQUALS a 2 3 "")

  lvalue(a.b)
  ans(res)
  assert(${res} EQUALS a 0 * b)

  lvalue(a.b[2])
  ans(res)
  assert(${res} EQUALS a 2 3 b)

  lvalue(a.b[2-3])
  ans(res)
  assert(${res} EQUALS a 2 4 b)

  lvalue(a.b[1-*])
  ans(res)
  assert(${res} EQUALS a 1 * b)


  return()


  function(lvalue_set lvalue)


  endfunction()
  function(lvalue_get)

  endfunction()

  lvalue("{a:{b:{c:1}}}")

return()


  obj("{d:1,c:2}")
  ans(a)
  json_print(${a})


  map_path_set(a d)

  return()
  map_path_set(a b c d e)
  map_path_set(a [] c)
  map_path_set(x y [2] d)

  message("${a}")
return()



  define_test_function(test_uut map_path_set)


  #test_uut("asd" thevar asd --print)
  #test_uut("{a:'b'}" thevar a b)
  #test_uut("1" thevar [] 1)







  return()

  function(map_path_append first)
    set(args ${ARGN})

    list_pop_back(args)
    ans(value)

    if(NOT args)
      set(${first} ${${first}} ${value} PARENT_SCOPE)
      return()
    endif()

    set(current ${${first}})

    if(NOT current)


    endif()


  endfunction()
endfunction()