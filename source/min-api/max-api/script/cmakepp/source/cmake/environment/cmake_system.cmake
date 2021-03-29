


function(cmake_system)
  cmake_check_configure_mode()

  map_new()
  ans(sys)

  map_set(${sys} id "${CMAKE_SYSTEM}")
  map_set(${sys} name "${CMAKE_SYSTEM_NAME}")
  map_set(${sys} version "${CMAKE_SYSTEM_VERSION}")
  map_set(${sys} processor "${CMAKE_SYSTEM_PROCESSOR}")


  return(${sys})
endfunction()