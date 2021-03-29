
# extracts the specified values from the command line (see list extract)
# returns the rest of the command line
# the first three arguments of commandline_get are cmake command, -P, script file 
# these are ignored
function(commandline_extract)
  commandline_get()
  ans(args)
  list_extract(args cmd p script ${ARGN})
  ans(res)
  vars_elevate(${ARGN})
  set(res ${cmd} ${p} ${script} ${res})
  return_ref(res)
endfunction()

