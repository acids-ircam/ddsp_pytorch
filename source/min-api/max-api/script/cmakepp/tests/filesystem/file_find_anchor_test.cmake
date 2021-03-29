function(test)



  pushd(dir1 --create)
    fwrite(".myanchor" "")
    pushd(dir2 --create)
      file_find_anchor(".myanchor")
      ans(anchor1)
    popd()
    pushd(dir3 --create)
      fwrite(".myanchor" "")
      file_find_anchor(".myanchor")
      ans(anchor2)
    
      file_find_anchor(".myanchor" "../")
      ans(anchor4)
    popd()



    file_find_anchor(".myanchor")
    ans(anchor3)
  
    file_find_anchor(".myanchor" "dir2")
    ans(anchor5)

    file_find_anchor(".myanchor" "dir3")
    ans(anchor6)
  popd()

  file_find_anchor(".myanchorabscscscs")
  ans(anchor7)

  assert("${anchor1}" STREQUAL "${test_dir}/dir1/.myanchor" )
  assert("${anchor2}" STREQUAL "${test_dir}/dir1/dir3/.myanchor" )
  assert("${anchor3}" STREQUAL "${test_dir}/dir1/.myanchor" )
  assert("${anchor4}" STREQUAL "${test_dir}/dir1/.myanchor" )
  assert("${anchor5}" STREQUAL "${test_dir}/dir1/.myanchor" )
  assert("${anchor6}" STREQUAL "${test_dir}/dir1/dir3/.myanchor" )
  assert("${anchor7}" ISNULL)



endfunction()