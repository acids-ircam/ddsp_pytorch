# compiles a tool (single cpp file with main method)
# and create a cmake function (if the tool is not yet compiled)
# expects tool to print cmake code to stdout. this code will 
# be evaluated and the result is returned  by the tool function
# the tool function's name is name
# currently only allows default headers
function(compile_tool name src)
  checksum_string("${src}")
  ans(chksum)

  cmakepp_config(temp_dir)
  ans(temp_dir)


  set(dir "${temp_dir}/tools/${chksum}")

  if(NOT EXISTS "${dir}")

    pushd("${dir}" --create)
    fwrite("main.cpp" "${src}")
    fwrite("CMakeLists.txt" "
      project(${name})
      if(\"\${CMAKE_CXX_COMPILER_ID}\" STREQUAL \"GNU\")
        include(CheckCXXCompilerFlag)
        CHECK_CXX_COMPILER_FLAG(\"-std=c++11\" COMPILER_SUPPORTS_CXX11)
        CHECK_CXX_COMPILER_FLAG(\"-std=c++0x\" COMPILER_SUPPORTS_CXX0X)
        if(COMPILER_SUPPORTS_CXX11)
          set(CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS} -std=c++11\")
        elseif(COMPILER_SUPPORTS_CXX0X)
          set(CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS} -std=c++0x\")
        else()
                message(STATUS \"The compiler \${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.\")
        endif()

      endif()
      set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   \${CMAKE_BINARY_DIR}/bin)
      set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE \${CMAKE_BINARY_DIR}/bin)
      set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   \${CMAKE_BINARY_DIR}/lib)
      set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE \${CMAKE_BINARY_DIR}/lib)
      set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG   \${CMAKE_BINARY_DIR}/lib)
      set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE \${CMAKE_BINARY_DIR}/lib)
      set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY \${CMAKE_BINARY_DIR}/lib)
      set(CMAKE_LIBRARY_OUTPUT_DIRECTORY \${CMAKE_BINARY_DIR}/lib)
      set(CMAKE_RUNTIME_OUTPUT_DIRECTORY \${CMAKE_BINARY_DIR}/bin)
      add_executable(${name} main.cpp)
      ")
    mkdir(build)
    cd(build)
    cmake(../ --process-handle)
    ans(configure_result)
    cmake(--build . --process-handle)
    ans(build_result)


    map_tryget(${build_result} exit_code)
    ans(error)
    map_tryget(${build_result} stdout)
    ans(log)
    popd()

    if(NOT "${error}" STREQUAL "0")        
      message(FATAL_ERROR "failed to compile tool :\n ${log}")
      rm("${dir}")
    endif()


  endif()
  
        
  wrap_executable_bare("__${name}" "${dir}/build/bin/${name}")

  eval("
    function(${name})

      __${name}(\${ARGN})
      ans_extract(error)
      if(error)
        message(FATAL_ERROR \"${name} tool (${dir}/build/bin/${name}) failed with \${error}\")
      endif()
      ans(stdout)
      eval(\"\${stdout}\")
     # _message(\${__ans})
      return_ans()
    endfunction()
    ")



endfunction()