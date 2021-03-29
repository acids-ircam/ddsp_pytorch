# returns a <process handle>
# currently does not play well with arguments
function(win32_wmic_call_create command)
  path("${command}")
  ans(cmd)
  pwd()
  ans(cwd)  
  set(args)


  message("cmd ${cmd}")
  file(TO_NATIVE_PATH "${cwd}" cwd)
  file(TO_NATIVE_PATH "${cmd}" cmd)


  if(ARGN)
    string(REPLACE ";" " " args "${ARGN}")
    set(args ",${args}")
  endif()
  win32_wmic(process call create ${cmd},${cwd})#${args}
  ans(res)
  set(pidregex "ProcessId = ([1-9][0-9]*)\;")
  set(retregex "ReturnValue = ([0-9]+)\;")
  string(REGEX MATCH "${pidregex}" pid_match "${res}")
  string(REGEX MATCH "${retregex}" ret_match "${res}")

  string(REGEX REPLACE "${retregex}" "\\1" ret "${ret_match}")
  string(REGEX REPLACE "${pidregex}" "\\1" pid "${pid_match}")
  if(NOT "${ret}" EQUAL 0)
    return()
  endif() 
  process_handle(${pid})
  ans(res)
  map_set(${res} status running)
  return_ref(res)
endfunction()