## `()-> <environment>`
##
## ```
## <environment descriptor> ::= {
##  host_name: <string> # Computer Name
##  processor: <string> # processor identification string
##  architecture: "32"|"64" # processor architecture
##  os:<operating system descriptor>
## }
## <operating system descriptor> ::= {
##  name: <string>
##  version: <string>
##  family: "Windows"|"Unix"|"MacOS"|...  
## }
## ```
## 
## returns the environment of cmake
## the results are cached (--update-cache if necesssary)
function(cmake_environment)
  function(_cmake_environment_inner)
  

    set(result)
    cmake_configure_script("map_capture_new(
      CMAKE_GENERATOR
      CMAKE_SIZEOF_VOID_P
      CMAKE_SYSTEM
      CMAKE_SYSTEM_NAME
      CMAKE_SYSTEM_PROCESSOR
      CMAKE_SYSTEM_VERSION
      CMAKE_HOST_SYSTEM
      CMAKE_HOST_SYSTEM_NAME
      CMAKE_HOST_SYSTEM_PROCESSOR
      CMAKE_HOST_SYSTEM_VERSION
      CMAKE_C_COMPILER_ID
      CMAKE_C_COMPILER_VERSION
      CMAKE_CXX_COMPILER_ID
      CMAKE_CXX_COMPILER_VERSION
      )")
    ans(res)
    
    site_name(host_name)
    assign(!result.host_name = host_name)     
    assign(!result.architecture = res.CMAKE_HOST_SYSTEM_PROCESSOR) 
    assign(!result.cmake.default_generator = res.CMAKE_GENERATOR)
    assign(!result.cmake.default_compiler_id = res.CMAKE_CXX_COMPILER_ID)
    assign(!result.cmake.default_compiler_version = res.CMAKE_CXX_COMPILER_VERSION)
  #  map_tryget(${res} CMAKE_SIZEOF_VOID_P)
   # ans(byte_size_voidp)
   # math(EXPR architecture "${byte_size_voidp} * 8")
   # assign(!result.architecture = architecture)
    
    assign(!result.os.name = res.CMAKE_HOST_SYSTEM_NAME)   
    assign(!result.os.version = res.CMAKE_HOST_SYSTEM_VERSION) 
    if(WIN32)
      assign(!result.os.family = 'Windows')   
    elseif(MAC)
      assign(!result.os.family = 'MacOS')   
    elseif(UNIX)
      assign(!result.os.family = 'Unix')  
    endif()
    return_ref(result)
  endfunction()
  
  define_cache_function(_cmake_environment_inner => cmake_environment
    --generate-key "[]()checksum_string({{CMAKE_COMMAND}})"
   # --refresh
  )
  cmake_environment(${ARGN})
  return_ans()
endfunction()

