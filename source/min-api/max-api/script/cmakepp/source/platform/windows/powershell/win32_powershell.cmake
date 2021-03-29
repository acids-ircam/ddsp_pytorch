
  ## wraps the win32 powershell command
  function(win32_powershell)
    wrap_executable(win32_powershell PowerShell)
    win32_powershell(${ARGN})
    return_ans()
  endfunction()
