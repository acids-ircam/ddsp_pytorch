function(test)
  ## parses a value descriptor


  value_descriptor_parse("test")
  ans(res)

  assert(res)
  assert(DEREF EQUALS "{res.id}" test)
  assert(DEREF EQUALS "{res.labels}" test)
  assert(DEREF EQUALS "{res.displayName}" "test")
  assert(DEREF EQUALS "{res.min}" 0)
  assert(DEREF EQUALS "{res.max}" 1)




  value_descriptor_parse("option1" LABELS --option1 -o1 MIN 0 MAX 0 DESCRIPTION "a single option" DISPLAY_NAME "Option Number One" DEFAULT test)
  ans(res)
  


  assert(DEREF EQUALS "{res.id}" option1)
  assert(DEREF EQUALS "{res.labels}" --option1 -o1)
  assert(DEREF EQUALS "{res.displayName}" "Option Number One")
  assert(DEREF EQUALS "{res.description}" "a single option")
  assert(DEREF EQUALS "{res.min}" 0)
  assert(DEREF EQUALS "{res.max}" 0)
  assert(DEREF EQUALS "{res.default}" test)


endfunction()