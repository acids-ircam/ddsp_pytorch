## wraps the windows task lisk programm which returns process info
function(win32_tasklist)
  wrap_executable(win32_tasklist "tasklist")
  win32_tasklist(${ARGN})
  return_ans()
endfunction()
