

function(cmake_capabilities)
  cmake(-E capabilities)
  ans(res)
  json_deserialize("${res}")
  ans(res)
  return_ref(res)
endfunction()