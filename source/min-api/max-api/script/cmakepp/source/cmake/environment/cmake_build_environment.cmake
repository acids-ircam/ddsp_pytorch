
 parameter_definition(cmake_build_environment)
function(cmake_build_environment)
  arguments_extract_defined_values(0 ${ARGC} cmake_build_environment)    
  ans(args)
  if(args AND NOT "${CMAKE_GENERATOR}" STREQUAL "${args}")
    cmake_configure_script("cmake_build_environment()" -G ${args})
    ans(res)
    return(${res})
  endif()

  cmake_check_configure_mode()
  cmake_compiler(CXX)
  ans(cxx)
  cmake_compiler(C)
  ans(c)


  map_capture_new(cxx c)
  ans(compilers)


  map_new()
  ans(env)

  cmake_system()
  ans(sys)


  # if(CMAKE_CROSSCOMPILING)
  #   map_set(${env} is_crosscompiling "true")
  # else()
  #   map_set(${env} is_crosscompiling "false")
  # endif()



  set(architecture "${CMAKE_SIZEOF_VOID_P}")
  math(EXPR architecture "${architecture} * 8")




  set(cfg "${CMAKE_BUILD_TYPE}")
  if(NOT cfg)
    set(cfg release)
  endif()

  if(DEFINED BUILD_SHARED_LIBS)
    
    if(BUILD_SHARED_LIBS)
      set(lnk shared)
    else()    
      set(lnk static)
    endif()
  else()
    set(lnk shared)
  endif()

  
  map_set(${env} generator ${CMAKE_GENERATOR})
  map_set(${env} config "${cfg}")
  map_set(${env} linkage "${lnk}")
  map_set(${env} architecture "${architecture}")
  map_set(${env} system "${sys}")
  map_set(${env} compilers ${compilers})



  return(${env})
endfunction()