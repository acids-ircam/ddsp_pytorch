function(test)


   semver_higher("2.0.1" "2.0.2")
   ans(res)
   assert("${res}" STREQUAL "2.0.2")

   semver_higher("2.0.3" "2.0.2")
   ans(res)
   assert("${res}" STREQUAL "2.0.3")

endfunction()