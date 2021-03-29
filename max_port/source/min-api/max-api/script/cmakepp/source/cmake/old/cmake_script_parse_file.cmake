
function(cmake_script_parse_file path)
  fread("${path}")
  ans(content)
  cmake_script_parse("${content}" ${ARGN})
  return_ans()
endfunction()