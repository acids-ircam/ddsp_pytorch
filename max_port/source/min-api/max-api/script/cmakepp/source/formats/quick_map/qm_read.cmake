# reads the qualifies and reads the specified <unqualified path>
# returns a <map>
function(qm_read path)
  path("${path}")
  ans(path)

  qm_deserialize_file("${path}")
  return_ans()
  
endfunction()