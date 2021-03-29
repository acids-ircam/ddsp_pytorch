function(test)

   semver_parse("0.331.21-alpha.dadad.23.123.asd+lolo.23.asd")
   ans(version)
   assert(version)

   semver_parse("0")
   ans(version)
   assert(NOT version)

   semver_parse("0.1")
   ans(version)
   assert(NOT version)

   semver_parse("0.1.1")
   ans(version)
   assert(version)
endfunction()