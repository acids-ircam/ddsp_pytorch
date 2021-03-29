## a bare wrapper for tasklist
function(win32_tasklist_bare)
  wrap_executable_bare(win32_tasklist_bare "tasklist")
  win32_tasklist_bare(${ARGN})
  return_ans()
endfunction()
