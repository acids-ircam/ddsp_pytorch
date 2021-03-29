function(test)

  semver_constraint_evaluate_element("=2.0.0" "2.0.0")
  ans(res)
  assert(res)
  
  semver_constraint_evaluate_element("=2" "2.0.0")
  ans(res)
  assert(res)
  
  semver_constraint_evaluate_element("=2.0" "2.0.0")
  ans(res)
  assert(res)
  
  semver_constraint_evaluate_element("2" "2.0.0")
  ans(res)
  assert(res)
  
  semver_constraint_evaluate_element("2" "2.0.1")
  ans(res)
  assert(NOT res)

  semver_constraint_evaluate_element("!2.0" "2.0.0")
  ans(res)
  assert(NOT res)
  
  semver_constraint_evaluate_element("!2.0" "2.0.0-alpha")
  ans(res)
  assert( res)

  semver_constraint_evaluate_element(">2" "2.0.1")
  ans(res)
  assert(res)
  
  semver_constraint_evaluate_element(">2" "2.0.0")
  ans(res)
  assert(NOT res)


  semver_constraint_evaluate_element("<2" "1.9.9")
  ans(res)
  assert(res)
 
  semver_constraint_evaluate_element("<2" "2.0.0")
  ans(res)
  assert(NOT res)

  semver_constraint_evaluate_element("~2.3.4" "2.0.1")
  ans(res)
  assert(NOT res)

  semver_constraint_evaluate_element("~2.3.4" "2.3.4")
  ans(res)
  assert(res)

  semver_constraint_evaluate_element("~2.3" "2.3.4")
  ans(res)
  assert(res)

  semver_constraint_evaluate_element("~2.3" "2.4.4")
  ans(res)
  assert(NOT res)

  semver_constraint_evaluate_element( "~2" "2.3.4")
  ans(res)
  assert(res)

  semver_constraint_evaluate_element( "~2" "3.0.0")
  ans(res)
  assert(NOT res)

  semver_constraint_evaluate_element( "~2" "2.0.0")
  ans(res)
  assert(res)

  semver_constraint_evaluate_element( "~2" "1.9.9")
  ans(res)
  assert(NOT res)

endfunction()