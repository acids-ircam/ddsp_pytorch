function(test)



map()
 kv(file1 content1)
 map(dir1)
  kv(file2 content2)
  kv(file3 content3)
  map(dir11)
   kv(file4 content4)
   kv(file5 content5)
   map(dir111)
  # kv(file9 content9)
   end()
  end()
end()
map(dir2)
 kv(file6 content6)
 kv(file7 content7)
 map(dir21)
  kv(file8 content8)
 end()
end()
end()
ans(fm)


pushd("test" --create)
file_map_write(${fm})
popd()



assert(EXISTS "${test_dir}/test/file1")
assert(IS_DIRECTORY "${test_dir}/test/dir1")
assert(EXISTS "${test_dir}/test/dir1/file2")
assert(EXISTS "${test_dir}/test/dir1/file3")
assert(IS_DIRECTORY "${test_dir}/test/dir1/dir11")
assert(EXISTS "${test_dir}/test/dir1/dir11/file4")
assert(EXISTS "${test_dir}/test/dir1/dir11/file5")
assert(IS_DIRECTORY "${test_dir}/test/dir2")
assert(EXISTS "${test_dir}/test/dir2/file6")
assert(EXISTS "${test_dir}/test/dir2/file7")
assert(IS_DIRECTORY "${test_dir}/test/dir2/dir21")
assert(EXISTS "${test_dir}/test/dir2/dir21/file8")

pwd()
ans(p)
message("pwd ${p}")
file_map_read(test)
ans(res)



json_print(${res})

endfunction()