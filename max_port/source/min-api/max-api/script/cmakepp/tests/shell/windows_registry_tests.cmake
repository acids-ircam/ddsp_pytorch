function(test)

  if(NOT WIN32)
    message("test inconclusive: windows only")
    return()
  endif()


  set(kv HKCU/Environment testValue1)

  ## read/write
  reg_write_value(${kv} "b;c")
  reg_read_value(${kv})
  ans(res)
  assert(EQUALS ${res} b c)

  ## append
  reg_append_value(${kv} "d")
  reg_read_value(${kv})
  ans(res)
  assert(EQUALS ${res} b c d)

  ## prepend
  reg_prepend_value(${kv} "a")
  reg_read_value(${kv})
  ans(res)
  assert(EQUALS ${res} a b c d)


  ## append if not exists
  reg_append_if_not_exists(${kv} b c e f)
  ans(res)
  assert(res)
  assert(EQUALS ${res} e f)
  reg_read_value(${kv})
  ans(res)
  assert(EQUALS ${res} a b c d e f)


  ## remove
  reg_remove_value(${kv} b d f)
  reg_read_value(${kv})
  ans(res)
  assert(EQUALS ${res} a c e)


  ## contains
  reg_contains_value(${kv} e)  
  ans(res)
  assert(res)


  ## read key
  reg_query_values(HKCU/Environment)
  ans(res)
  json_print(${res})
  assert(EQUALS DEREF {res.testValue1} a c e)







endfunction()
