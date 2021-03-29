function(test)

pushd(archive1 --create)
    fwrite("asd.txt" "asd")
    fwrite("needle.cmake" "{}")
    fwrite("bsd.cmake" "{}")
    fwrite("dir/bsd.needle.cmake" "{}")
compress("../archive1.tgz" "**")

popd()


pushd(archive2 --create)
    fwrite("asd.txt" "asd")
    fwrite("dir/needle.cmake" "{}")
    fwrite("bsd.cmake" "{}")
    fwrite("dir/bsd.needle.cmake" "{}")
compress("../archive2.tgz" "**")
popd()



  archive_match_files("archive1.tgz" "([^;]+/)?needle\\.cmake")
  ans(res)
  assert("${res}" STREQUAL "needle.cmake")

  archive_match_files("archive2.tgz" "([^;]+/)?needle\\.cmake")
  ans(res)
  assert("${res}" STREQUAL "dir/needle.cmake")
endfunction()