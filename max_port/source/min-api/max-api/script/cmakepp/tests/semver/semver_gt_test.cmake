function(test)


   semver_gt("3.0.0" "2.0.0")
   ans(res)
   assert(res)
   semver_gt("3.0.0" "4.0.0")
   ans(res)
   assert(NOT res)

endfunction()