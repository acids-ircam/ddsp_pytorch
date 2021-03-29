function(test)

# splits a list into two parts before index 3
set(lst 1 2 3 4 5 6 7)
list_split(p1 p2 lst 3)
list_to_string( p1 " ")
ans(p1)
list_to_string( p2 " ")
ans(p2)
assert("1 2 3" STREQUAL "${p1}")
assert("4 5 6 7" STREQUAL "${p2}")

endfunction()