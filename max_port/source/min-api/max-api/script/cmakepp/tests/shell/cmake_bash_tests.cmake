function(test)
  
  
  path("${test_dir}")
   ans(p)
  assert("${p}" STREQUAL "${test_dir}")
  if(NOT EXISTS "${test_dir}")
    file(MAKE_DIRECTORY "${test_dir}")
  endif()


  cd("${test_dir}")
  ans(res)
  assert(EXISTS "${res}")

  pwd()
  ans(res)
  assert("${res}" STREQUAL "${test_dir}")

  path(dir1)
  ans(res)
  assert("${res}" STREQUAL "${test_dir}/dir1")

  file(MAKE_DIRECTORY "${test_dir}/dir1")

  cd(dir1)
  ans(res)
  assert("${res}" STREQUAL "${test_dir}/dir1")


  pwd()
  ans(res)
  assert("${res}" STREQUAL "${test_dir}/dir1")

  cd(..)
  ans(res)
  assert("${res}" STREQUAL "${test_dir}")

  mkdir(mydir)
  ans(res)
  assert("${test_dir}/mydir" STREQUAL "${res}")
  assert(EXISTS "${test_dir}/mydir")

pwd()
ans(pwd)

assert("${pwd}" STREQUAL "${test_dir}")

path(test)
ans(res)
assert("${res}" STREQUAL "${test_dir}/test")

  touch(test2)
  assert(EXISTS "${test_dir}/test2")

# seems to cause problems
#  dirs()
#  ans(res)
#  assert(NOT res)
#  pushd()
#  dirs()
#  ans(res)
#  assert("${res}" STREQUAL "${test_dir}")
#  cd(mydir)
#  pwd()
#  ans(res)
#  assert("${res}" STREQUAL "${test_dir}/mydir")
#  popd()
#  pwd()
#  ans(res)
#  assert("${res}" STREQUAL "${test_dir}")
#


  touch(test1/f1.txt)
  touch(test1/f2.txt)
  ls(test1)
  ans(res)

  message("ls ${res}")

assert(COUNT 2 ${res})  

endfunction()
