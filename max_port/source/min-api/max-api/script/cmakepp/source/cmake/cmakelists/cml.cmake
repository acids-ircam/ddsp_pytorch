## `(...)->...`
## 
## wrapper for cmakelists_cli
function(cml)
  set(args ${ARGN})
  cmakelists_cli(${args})
  ans(res)
  return_ref(res)
endfunction()
