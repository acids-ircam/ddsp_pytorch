function(test)

semver_constraint_compile("some bogues constraint 0.0.1|>0.1")
ans(res)
assert(NOT res)

semver_constraint_compile(">1|<2")
ans(res)
assert(res)
assert(DEREF EQUALS "{res.template}" >1 OR <2)
assert(DEREF EQUALS "{res.elements}" >1 <2)







    
endfunction()