## sets up the cmakepp environment 
## creates aliases
##    icmake - interactive cmakepp
##    cmakepp - commandline interface to cmakepp 
##    pkg - package manager command line interface
function(cmakepp_setup_environment)
  cmakepp_config(base_dir)
  ans(base_dir)


  
  message(STATUS "creating alias `icmakepp`")  
  alias_create("icmakepp" "cmake -P ${base_dir}/cmakepp.cmake icmake")
  message(STATUS "creating alias `cmakepp`")  
  alias_create("cmakepp" "cmake -P ${base_dir}/cmakepp.cmake")
  message(STATUS "creating alias `pkg`")  
  alias_create("pkg" "cmake -P ${base_dir}/cmakepp.cmake cmakepp_project_cli")
  message(STATUS "creating alias `cml`")  
  alias_create("cml" "cmake -P ${base_dir}/cmakepp.cmake cmakelists_cli")
  message(STATUS "setting CMAKEPP_PATH to ${base_dir}/cmakepp.cmake ")

  shell_env_set(CMAKEPP_PATH "${base_dir}/cmakepp.cmake")


  
endfunction()