# writes the specified values to path as a quickmap file
# path is an <unqualified file>
# returns the <qualified path> were values were written to
function(qm_write path )
  path("${path}")
  ans(path)

  qm_serialize(${ARGN})
  ans(res)
  fwrite("${path}" "${res}")
  return_ref(path)
endfunction()