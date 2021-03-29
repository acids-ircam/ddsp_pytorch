
function(win32_cmd_lean)
  wrap_executable_bare(win32_cmd_lean cmd.exe)
  win32_cmd_lean(${ARGN})
  return_ans()
endfunction()