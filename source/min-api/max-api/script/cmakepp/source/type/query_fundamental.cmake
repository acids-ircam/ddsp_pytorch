
  function(query_fundamental input_callback type)
      
      call("${input_callback}"(${type}))
      ans(res)
      return_ref(res)
  endfunction()