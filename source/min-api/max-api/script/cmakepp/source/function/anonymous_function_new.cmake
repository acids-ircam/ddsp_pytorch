function(anonymous_function_new code)
  function_new()
  ans(function_name)
  set(function_name "anonymous_${function_name}")
  string(REGEX REPLACE "^\\(" "(${function_name}" code "${code}")
  eval("function${code}endfunction()")
  return_ref(function_name)
endfunction()

