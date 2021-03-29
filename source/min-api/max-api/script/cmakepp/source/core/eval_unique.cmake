## 
## 
## for special cases eval unique will create a new file for every code
## 
function(eval_unique __unique_eval_code)
  mk_temp()
  ans(dir)
  eval("
  function(eval_unique __unique_eval_code)
    string(MD5 file \"\${__unique_eval_code}\")
    file(WRITE \"${dir}/\${file}\" \"\${__unique_eval_code}\")
    include(\"${dir}/\${file}\")
  endfunction()
  ")
  eval_unique("${__unique_eval_code}")
  return_ans()
endfunction()


