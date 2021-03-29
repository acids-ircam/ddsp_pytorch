function(test)


  ## create 3 file data stores
  file_data_write_obj(dir1 mydata "{id:1, a:1}")
  file_data_write_obj(dir2 mydata "{id:2, b:2}")
  file_data_write_obj(dir3 mydata "{id:3, c:3}")

  ## act 
  fallback_data_read("dir1;dir2;dir3" mydata)
  ans(res)  

  #assert that the combined data is correct  
  # precedence left to right -> id is written to all thre
  map_equal_obj("${res}" "{id:1,a:1,b:2,c:3}")

  ## assert that invalid key returns no data
  fallback_data_get("dir1;dir2;dir3" mydata invalidkey)
  ans(res)
  assert(NOT res)

  ## assert that fallback_data_get returns the correct data
  fallback_data_get("dir1;dir2;dir3" mydata id)
  ans(res)
  assert("${res}" STREQUAL "1")

  ## assert that fallback_data_get returns the correct value definined in dir1
  fallback_data_get("dir1;dir2;dir3" mydata a)
  ans(res)
  assert("${res}" STREQUAL "1")

  ## assert that fallback_data_get returns the correct value for dir2
  fallback_data_get("dir1;dir2;dir3" mydata b)
  ans(res)
  assert("${res}" STREQUAL "2")

  ## assert that fallback_data_get returns the correct value for dir3
  fallback_data_get("dir1;dir2;dir3" mydata c)
  ans(res)
  assert("${res}" STREQUAL "3")

  ## assert that the correct data source is returned for key id
  fallback_data_source("dir1;dir2;dir3" mydata id)
  ans(res)
  assert("${res}" STREQUAL "dir1")

  ## assert that the correct data source is returned for key a
  fallback_data_source("dir1;dir2;dir3" mydata a)
  ans(res)
  assert("${res}" STREQUAL "dir1")

  ## assert that the correct data source is returned for key b
  fallback_data_source("dir1;dir2;dir3" mydata b)
  ans(res)
  assert("${res}" STREQUAL "dir2")

  ## assert that the correct data source is returned for key c
  fallback_data_source("dir1;dir2;dir3" mydata c)
  ans(res)
  assert("${res}" STREQUAL "dir3")

  ## assert that no data source is returned for invalid key
  fallback_data_source("dir1;dir2;dir3" mydata invalidkey)
  ans(res)
  assert(NOT res)

  


endfunction()


