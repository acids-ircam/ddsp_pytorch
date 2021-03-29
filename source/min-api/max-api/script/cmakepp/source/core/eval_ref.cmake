# macro version of eval function which causes set(PARENT_SCOPE ) statements to access 
# scope of invokation
macro(eval_ref __eval_code_ref)
  fwrite_temp("" ".cmake")
  eval("
 macro(eval_ref __eval_code_ref_inner)
   file(WRITE ${__ans} \"\${\${__eval_code_ref_inner}}\")
   include(${__ans})
 endmacro()
   ")
 eval_ref(${__eval_code_ref})
endmacro()


