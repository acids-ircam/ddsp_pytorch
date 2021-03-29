function(test)

  ## {
  ##   input: <string>
  ##   uri: <uri string>
  ##   base_uri: <uri string>
  ##   relative_uri: <path>
  ##   ref_type: "branch"|"tag"|"trunk"
  ##   ref: <string>
  ##   revision: <rev>
  ## }

  svn_uri_analyze("https://github.com/toeb/test_repo/trunk/asdbsd/sd" --revision 3)
  ans(res)

  assertf("{res.input}" STREQUAL "https://github.com/toeb/test_repo/trunk/asdbsd/sd")
  assertf("{res.uri}" STREQUAL "https://github.com/toeb/test_repo/trunk/asdbsd/sd")
  assertf("{res.base_uri}" STREQUAL "https://github.com/toeb/test_repo")
  assertf("{res.relative_uri}" STREQUAL "asdbsd/sd")
  assertf("{res.ref_type}" STREQUAL "trunk")
  assertf("{res.ref}" STREQUAL "trunk")
  assertf("{res.revision}" STREQUAL "3")



  svn_uri_analyze("https://github.com/toeb/test_repo@2")
  ans(res)

  assertf("{res.input}" STREQUAL "https://github.com/toeb/test_repo@2")
  assertf("{res.uri}" STREQUAL "https://github.com/toeb/test_repo")
  assertf("{res.base_uri}" STREQUAL "https://github.com/toeb/test_repo")
  assertf("{res.relative_uri}" ISNULL)
  assertf("{res.ref_type}" STREQUAL "trunk")
  assertf("{res.ref}" ISNULL)
  assertf("{res.revision}" STREQUAL "2")


  svn_uri_analyze("https://github.com/toeb/test_repo" --revision 3)
  ans(res)

  assertf("{res.input}" STREQUAL "https://github.com/toeb/test_repo")
  assertf("{res.uri}" STREQUAL "https://github.com/toeb/test_repo")
  assertf("{res.base_uri}" STREQUAL "https://github.com/toeb/test_repo")
  assertf("{res.relative_uri}" ISNULL)
  assertf("{res.ref_type}" STREQUAL "trunk")
  assertf("{res.ref}" ISNULL)
  assertf("{res.revision}" STREQUAL "3")


  svn_uri_analyze("https://github.com/toeb/test_repo@2" --revision 3)
  ans(res)

  assertf("{res.input}" STREQUAL "https://github.com/toeb/test_repo@2")
  assertf("{res.uri}" STREQUAL "https://github.com/toeb/test_repo")
  assertf("{res.base_uri}" STREQUAL "https://github.com/toeb/test_repo")
  assertf("{res.relative_uri}" ISNULL)
  assertf("{res.ref_type}" STREQUAL "trunk")
  assertf("{res.ref}" ISNULL)
  assertf("{res.revision}" STREQUAL "3")


  svn_uri_analyze("https://github.com/toeb/test_repo/trunk" --revision 3)
  ans(res)

  assertf("{res.input}" STREQUAL "https://github.com/toeb/test_repo/trunk")
  assertf("{res.uri}" STREQUAL "https://github.com/toeb/test_repo/trunk")
  assertf("{res.base_uri}" STREQUAL "https://github.com/toeb/test_repo")
  assertf("{res.relative_uri}" ISNULL)
  assertf("{res.ref_type}" STREQUAL "trunk")
  assertf("{res.ref}" STREQUAL "trunk")
  assertf("{res.revision}" STREQUAL "3")


  svn_uri_analyze("https://github.com/toeb/test_repo/branches/b1" --revision 3)
  ans(res)

  assertf("{res.input}" STREQUAL "https://github.com/toeb/test_repo/branches/b1")
  assertf("{res.uri}" STREQUAL "https://github.com/toeb/test_repo/branches/b1")
  assertf("{res.base_uri}" STREQUAL "https://github.com/toeb/test_repo")
  assertf("{res.relative_uri}" ISNULL)
  assertf("{res.ref_type}" STREQUAL "branch")
  assertf("{res.ref}" STREQUAL "b1")
  assertf("{res.revision}" STREQUAL "3")


  svn_uri_analyze("https://github.com/toeb/test_repo/tags/b1" --revision 3)
  ans(res)

  assertf("{res.input}" STREQUAL "https://github.com/toeb/test_repo/tags/b1")
  assertf("{res.uri}" STREQUAL "https://github.com/toeb/test_repo/tags/b1")
  assertf("{res.base_uri}" STREQUAL "https://github.com/toeb/test_repo")
  assertf("{res.relative_uri}" ISNULL)
  assertf("{res.ref_type}" STREQUAL "tag")
  assertf("{res.ref}" STREQUAL "b1")
  assertf("{res.revision}" STREQUAL "3")


  svn_uri_analyze("https://github.com/toeb/test_repo/tags/b1" --revision 3 --tag asd)
  ans(res)

  assertf("{res.input}" STREQUAL "https://github.com/toeb/test_repo/tags/b1")
  assertf("{res.uri}" STREQUAL "https://github.com/toeb/test_repo/tags/b1")
  assertf("{res.base_uri}" STREQUAL "https://github.com/toeb/test_repo")
  assertf("{res.relative_uri}" ISNULL)
  assertf("{res.ref_type}" STREQUAL "tag")
  assertf("{res.ref}" STREQUAL "asd")
  assertf("{res.revision}" STREQUAL "3")

endfunction()