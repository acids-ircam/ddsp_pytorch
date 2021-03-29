function(test)

 
  
  # single multivalue
  value_descriptor_parse("test1" LABELS "--o1" MIN 0 MAX 0)
  ans(descriptor)


  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused "--o1") 
  ans(res)
  assert(res)
  assert(NOT unused)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused hello there )
  ans(res)
  assert(NOT res)
  assert(EQUALS ${unused} hello there)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused value value2 --o1 value3 )
  ans(res)
  assert(res)
  assert(EQUALS ${unused} value value2 value3)

  # single single value 
  value_descriptor_parse("test1" LABELS "--o1" MIN 0 MAX 1)
  ans(descriptor)


  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused --o1)
  ans(res)
  assert(NOT res)
  assert(NOT unused )


  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused )
  ans(res)
  assert(NOT res)
  assert(NOT unused)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused asd bsd)
  ans(res)
  assert(NOT res)
  assert(EQUALS ${unused} asd bsd)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused --o1 "hello world")
  ans(res)
  assert(EQUALS ${res} "hello world" )
  assert(NOT unused)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused lala --o1 mumu haha)
  ans(res)
  assert(EQUALS ${res} "mumu")
  assert(EQUALS ${unused} lala haha)

  #single 0 * value 
  value_descriptor_parse("test1" LABELS "--o1" MIN 0 MAX *)
  ans(descriptor)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused --o1 a b c d)
  ans(res)
  assert(EQUALS ${res} a b c d)
  assert(NOT unused)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused a b --o1 c d e)
  ans(res)
  assert(EQUALS ${res} c d e)
  assert(EQUALS ${unused} a b)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused a b c d)
  ans(res)
  assert(NOT res)
  assert(EQUALS ${unused} a b c d)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused --o1)
  ans(res)
  assert(NOT res)
  assert(NOT unused)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused a b c --o1)
  ans(res)
  assert(NOT res)
  assert(EQUALS ${unused} a b c)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused)
  ans(res)
  assert(NOT res)
  assert(NOT unused)


  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused --o1 a b c)
  ans(res)
  assert(EQUALS ${res} a b c)
  assert(NOT unused)


  # single bounded multi value 2 4
  value_descriptor_parse("test1" LABELS "--o1" MIN 2 MAX 4)
  ans(descriptor)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused)
  ans(res)
  assert(error)
  assert(NOT res)
  assert(NOT unused)


  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused --o1)
  ans(res)
  assert(error)
  assert(NOT res)
  assert(NOT unused)

  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused --o1 a)
  ans(res)
  assert(error)
  assert(EQUALS ${res} a)
  assert(NOT unused)


  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused --o1 a b)
  ans(res)
  assert(NOT error)
  assert(EQUALS ${res} a b)
  assert(NOT unused)


  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused --o1 a b c)
  ans(res)
  assert(NOT error)
  assert(EQUALS ${res} a b c)
  assert(NOT unused)


  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused --o1 a b c d e)
  ans(res)
  assert(NOT error)
  assert(EQUALS ${res} a b c d)
  assert(EQUALS ${unused} e)


  list_parse_descriptor(${descriptor} ERROR error UNUSED_ARGS unused 1 2 3 --o1 a b c d e)
  ans(res)
  assert(NOT error)
  assert(EQUALS ${res} a b c d)
  assert(EQUALS ${unused} 1 2 3 e)


endfunction()