# Evaluate expression (faster version)
# Suggestion from the Wiki: http://cmake.org/Wiki/CMake/Language_Syntax
# Unfortunately, no built-in stuff for this: http://public.kitware.com/Bug/view.php?id=4034
# eval will not modify ans (the code evaluated may modify ans)
# vars starting with __eval should not be used in code
function(eval __eval_code)
  
  # one file per execution of cmake (if this file were in memory it would probably be faster...)
  fwrite_temp("" ".cmake")
  ans(__eval_temp_file)


# speedup: statically write filename so eval boils down to 3 function calls
 file(WRITE "${__eval_temp_file}" "
function(eval __eval_code)
  file(WRITE ${__eval_temp_file} \"\${__eval_code}\")
  include(${__eval_temp_file})
  set(__ans \${__ans} PARENT_SCOPE)
endfunction()
  ")
include("${__eval_temp_file}")


eval("${__eval_code}")
return_ans()
endfunction()


