## `(<subdir:<path>> <?path> )-><bool>`
##
## returns true iff subdir is or is below path
function(path_issubdir subdir path)
  set(path ${ARGN})
  path_qualify(path)
  path_qualify(subdir)
  string_starts_with("${subdir}" "${path}")
  return_ans()
endfunction()
