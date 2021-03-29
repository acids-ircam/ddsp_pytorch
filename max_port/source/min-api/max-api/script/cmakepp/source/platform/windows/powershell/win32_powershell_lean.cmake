## wraps the win32 powershell command in a lean wrapper
function(win32_powershell_lean)
  wrap_executable_bare(win32_powershell_lean PowerShell)
  win32_powershell_lean(${ARGN})
  return_ans()
endfunction()
