## can be used as a standalone file to get a working copy of cmakepp
function(cmakepp_require ) 
  set(installation_dir ${ARGN})
  if("${installation_dir}_" STREQUAL "_")
    set(installation_dir "${CMAKE_CURRENT_BINARY_DIR}/cmakepp")
  endif()

  ## prefer local verison
  if(EXISTS "${installation_dir}/cmakepp.cmake")
    message(STATUS "Using CMake++ from local installation")
    include("${installation_dir}/cmakepp.cmake")
    return()
  endif() 

  ## prefer version
  if(EXISTS "$ENV{CMAKEPP_PATH}")
    message(STATUS "Using CMake++ from path")
    include("$ENV{CMAKEPP_PATH}")
    return()
  endif()

  ## download cmakepp
  set(git_uri "https://github.com/AnotherFoxGuy/cmakepp")
  set(cmakepp_uri "${git_uri}/releases/download/v0.0.3/cmakepp.cmake")
  set(target_file "${CMAKE_CURRENT_BINARY_DIR}/__cmakepp.cmake")

  message(STATUS "Installing CMake++")
  message(STATUS "\n installation_dir: ${installation_dir}")


  file(DOWNLOAD "${cmakepp_uri}" "${target_file}" STATUS status)
  include("${target_file}")
  file(REMOVE "${target_file}")

  ## uses git functionality of cmakepp v0.0.3 
  ## to download the current git repository
  git(clone "${git_uri}.git" "${installation_dir}")

  cmake(-P "${installation_dir}/cmakepp.cmake" cmakepp_compile "${installation_dir}/tmp/cmakepp.cmake") 
  file(READ "${installation_dir}/tmp/cmakepp.cmake")
  rm(-r "${installation_dir}")
  file(WRITE "${installation_dir}/cmakepp.cmake")

 
## so this line causes a segmentation fault so i'm just gonna ignore it for now...
 ## include("${installation_dir}/cmakepp.cmake")

  return()
endfunction()