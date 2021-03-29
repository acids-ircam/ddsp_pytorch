function(test)
set(thelist 1 2 3 4 5 6)

#list slice tests
list_slice(thelist 0 1)
ans(res)
assert(EQUALS ${res} 1 )

list_slice(thelist 1 2)
ans(res)
assert(EQUALS ${res} 2)

list_slice(thelist 5 6)
ans(res)
assert(EQUALS 6 ${res})

list_slice(thelist 0 6)
ans(res)
assert(EQUALS ${res} ${thelist})

list_slice(thelist 2 5)
ans(res)
assert(EQUALS ${res} 3 4 5)


list_slice(thelist -2 -1)
ans(res)
assert(EQUALS ${res} 6)

list_slice(thelist -7 -6)
ans(res)
assert(EQUALS ${res} 1)

list_slice(thelist 1 1)
ans(res)
assert(NOT res)

list_slice(thelist -2 -2)
ans(res)
assert(NOT res)

list_slice(thelist 4 2)
ans(res)
assert(EQUALS ${res} 5 4 )





endfunction()