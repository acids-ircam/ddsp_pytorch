# wraps the win32 taskkill command
function(win32_taskkill)
  wrap_executable(win32_taskkill "taskkill")
  win32_taskkill(${ARGN})
  return_ans()
endfunction()