

function(cmake_host_system)
 
  cmake_is_configure_mode()
  ans(is_config_mode)
  if(NOT is_config_mode)
    cmake_configure_script("cmake_host_system()")
    return_ans()
  endif()


  map_new()
  ans(sys)

  map_set(${sys} id "${CMAKE_HOST_SYSTEM}")
  map_set(${sys} name "${CMAKE_HOST_SYSTEM_NAME}")
  map_set(${sys} version "${CMAKE_HOST_SYSTEM_VERSION}")
  map_set(${sys} processor "${CMAKE_HOST_SYSTEM_PROCESSOR}")


  return(${sys})
endfunction()