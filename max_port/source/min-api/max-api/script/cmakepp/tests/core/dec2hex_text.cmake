function(test)


dec2hex(195936478)
ans(res)
assert("${res}" STREQUAL "BADC0DE")


dec2hex(0)
ans(res)
assert("${res}" STREQUAL 0)

dec2hex(15)
ans(res)
assert("${res}" STREQUAL "F")



endfunction()