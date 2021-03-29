function(test)


  obj("{asd:123,bsd:323}")
  ans(res)
  assert(${res} MAP_MATCHES "{asd:123}")



function(some_test)
  return(a b c)
endfunction()
assert(a b c EQUALS CALL some_test())

# assert boolean value and continue 
assert(false MESSAGE "should be true"  RESULT res)
assert(NOT res)


#assert boolean value 
assert(true MESSAGE "should be true"  RESULT res)
assert(res)


# assert false truth expression
assert("asd" STREQUAL "bsd"  RESULT res)
assert(NOT res)

# assert true truth expression
assert("asd" STREQUAL "asd"  RESULT res)
assert(res)

# assert false equality
assert(EQUALS asd bsd  RESULT res )
assert(NOT res)

#assert true equality
assert(EQUALS asd asd  RESULT res)
assert(res)


## assert list equality true

# you can specify both lists inline
assert(EQUALS 1 2 3 4 1 2 3 4  RESULT res)
assert(res)

# you can specifiy the list a string
assert(EQUALS "1;2;3;4" "1;2;3;4"  RESULT res)
assert(res)

# you can pass the list by reference
set(listA 1 2 3 4)
set(listB 1 2 3 4)


assert(EQUALS listA listB  RESULT res)
assert(res)

## assert list equality false
assert(EQUALS 1 2 3 4 2 2 3 4  RESULT res)
assert(NOT res)

assert(EQUALS "1;2;3;4" "2;3;4;4"  RESULT res)
assert(NOT res)

set(listA 1 2 3 4)
set(listB 1 2 3 3)
assert(EQUALS listA listB  RESULT res)
assert(NOT res)




endfunction()