function(test)



hex2dec(AA8fdS)
ans(res)
assert(${res} ISNULL)

hex2dec(FFFF)
ans(res)
assert("${res}" EQUAL 65535)

hex2dec(badc0de)
ans(res)
assert("${res}" EQUAL 195936478)

hex2dec(badf00d)
ans(res)
assert("${res}" EQUAL 195948557)

hex2dec("")
ans(res)
assert("${res}" ISNULL)


hex2dec("k")
ans(res)
assert("${res}" ISNULL)


hex2dec(0)
ans(res)
assert("${res}" EQUAL 0)

hex2dec(1)
ans(res)
assert("${res}" EQUAL 1)

hex2dec(2)
ans(res)
assert("${res}" EQUAL 2)

hex2dec(3)
ans(res)
assert("${res}" EQUAL 3)

hex2dec(4)
ans(res)
assert("${res}" EQUAL 4)

hex2dec(5)
ans(res)
assert("${res}" EQUAL 5)

hex2dec(6)
ans(res)
assert("${res}" EQUAL 6)

hex2dec(7)
ans(res)
assert("${res}" EQUAL 7)

hex2dec(8)
ans(res)
assert("${res}" EQUAL 8)

hex2dec(9)
ans(res)
assert("${res}" EQUAL 9)

hex2dec(A)
ans(res)
assert("${res}" EQUAL 10)

hex2dec(B)
ans(res)
assert("${res}" EQUAL 11)

hex2dec(C)
ans(res)
assert("${res}" EQUAL 12)

hex2dec(D)
ans(res)
assert("${res}" EQUAL 13)

hex2dec(E)
ans(res)
assert("${res}" EQUAL 14)

hex2dec(F)
ans(res)
assert("${res}" EQUAL 15)


hex2dec(a)
ans(res)
assert("${res}" EQUAL 10)

hex2dec(b)
ans(res)
assert("${res}" EQUAL 11)

hex2dec(c)
ans(res)
assert("${res}" EQUAL 12)

hex2dec(d)
ans(res)
assert("${res}" EQUAL 13)

hex2dec(e)
ans(res)
assert("${res}" EQUAL 14)

hex2dec(f)
ans(res)
assert("${res}" EQUAL 15)


endfunction()