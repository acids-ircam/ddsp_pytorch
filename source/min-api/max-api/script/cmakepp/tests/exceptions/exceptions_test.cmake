function(test)

  function(testf)
    if(ARGN)
      message(mu)
      return(ok)
    else()
      message(hu)
      throw("damnit")
    endif()
  endfunction()


  testf(true)
  ans(res)
  is_exception("${res}")
  ans(is_exception)
  assert(NOT is_exception)



  testf(false)
  ans(res)
  assert(NOT res)
  is_exception("${res}")
  ans(is_exception)
  assert(is_exception)
  assertf("{res.message}" STREQUAL damnit)
  



  testf(true)
  catch((ex) return(was_exception))
  ans(res)
  assert("${res}" STREQUAL ok)


  testf(false)
  ans(result)
  catch((ex) return(was_exception))
  ans(res)
  assert(NOT result)
  assert(res)
  assert("${res}" STREQUAL "was_exception")


  ## exception thrown inside exception
  ## 

  testf(false)
  ans(ex1)
  catch((ex) throw (other_exception))
  ans(ex2)
  catch((ex) map_tryget(\${ex} message) ans(msg) print_vars(exception) print_vars(ex) print_vars(ex.message) return_ref(msg) )
  ans(res)

  assert(res)
  assert("${res}" STREQUAL other_exception)



  function(testf2)
    testf("${ARGN}")
    rethrow()
    ans(res)

    return("${res}ok")
  endfunction()


  testf2(true)
  ans(res)

  assert("${res}" STREQUAL okok)

  testf2(false)
  ans(res)
  catch((ex) print_vars(ex) map_tryget(\${ex} message) return_ans())
  ans(message)
  assert(NOT res)
  is_exception("${res}")
  ans(res)
  assert(res)

  assert("${message}" STREQUAL "damnit")






endfunction()