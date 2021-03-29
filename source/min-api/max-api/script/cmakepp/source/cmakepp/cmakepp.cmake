## 
## executes the cmakepp command line as a separate process
##
## 
function(cmakepp)
  cmakepp_config(base_dir)
  ans(base_dir)
  cmake("-P" "${base_dir}/cmakepp.cmake" ${ARGN})
  return_ans()    
endfunction()

