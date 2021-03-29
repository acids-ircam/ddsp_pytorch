# wraps the win32 console executable cmd.exe
function(win32_cmd)
  wrap_executable(win32_cmd cmd.exe)
  win32_cmd(${ARGN})
  return_ans()
endfunction()

