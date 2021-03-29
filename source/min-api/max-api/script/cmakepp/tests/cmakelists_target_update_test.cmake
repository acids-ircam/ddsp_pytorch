function(test)



  cmakelists_new("
    add_library(my_lib source1.c)
    target_link_libraries(my_lib lib2)
    add_library(my_lib2 source2.c )
    target_link_libraries(my_lib2 lib2)
    ")
  ans(cmakelists)

  cmakelists_target("${cmakelists}" my_lib2)
  ans(target)



  map_append(${target} target_source_files source3.c)
  map_set(${target} target_type executable)
  map_set(${target} target_link_libraries)

  cmakelists_target_update("${cmakelists}" "${target}")
  cmakelists_serialize("${cmakelists}")

  map_set(${target} target_link_libraries
   asdasd
   bsd
   csd 
   dsd
   kaka
   asdasd
   asdaisojda
   asdoijaosidja
   asoidjaoisjdiaojdiojasiodj
   )
  map_set(${target} target_include_directories incl 

    asdiaoshdoas
    asdoihaoisuhd
    asuidhoaiushd
    aisdoiauhsdioua
    aiosgduioagshdu)



  cmakelists_target_update("${cmakelists}" "${target}")

  map_new()
  ans(new_target)
  map_set(${new_target} target_name hihi_target)
  map_set(${new_target} target_type executable)
  map_set(${new_target} target_source_files src1 src2)

  cmakelists_target_update("${cmakelists}" ${new_target})


  cmakelists_serialize("${cmakelists}")
  ans(res)


  assert("${res}" MATCHES "aiosgduioagshdu")
  assert("${res}" MATCHES "hihi_target")



  cmakelists_new("${res}")
  ans(cmakelists)

  cmakelists_target(${cmakelists} "hihi_target")
  ans(target)


  map_set(${target} target_type library)
  map_set(${target} target_name other_target)

  cmakelists_target_update("${cmakelists}" "${target}")


  cmakelists_serialize("${cmakelists}")
  ans(res)
  cmakelists_new("${res}")
  ans(cmakelists)
  cmakelists_target(${cmakelists} "hihi_target")
  ans(target)
  assert(NOT target)

  cmakelists_target(${cmakelists} "other_target")
  ans(target)
  assert(target)




endfunction()