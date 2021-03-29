function(test)

  semver_constraint_evaluate("=0.0.1" "0.0.1") 
  ans(res)
  assert(res)
 
  semver_constraint_evaluate("=0.0.1" "0.0.2") 
  ans(res)
  assert(NOT res)
 
  semver_constraint_evaluate("!0.0.1" "0.0.1") 
  ans(res)
  assert(NOT res)
 
  semver_constraint_evaluate("!0.0.1" "0.0.2") 
  ans(res)
  assert(res)
 
  semver_constraint_evaluate(">0.0.1" "0.0.2") 
  ans(res)
  assert(res)

  semver_constraint_evaluate(">0.0.1" "0.0.1") 
  ans(res)
  assert(NOT res)
 
  semver_constraint_evaluate("<0.0.1" "0.0.0") 
  ans(res)
  assert(res)
  
  semver_constraint_evaluate("<0.0.1" "0.0.1") 
  ans(res)
  assert(NOT res)
 
  semver_constraint_evaluate("<=3,>2" "3.0.0")
  ans(res)
  assert(res)
 
  semver_constraint_evaluate("<=3,>=2" "2.0.0")
  ans(res)
  assert(res)

endfunction()