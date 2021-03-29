# parses the command line string into parts (handling strings and semicolons)
function(parse_command_line result args)

  string(ASCII  31 ar)
  string(REPLACE "\;" "${ar}" args "${args}" )
  string(REGEX MATCHALL "((\\\"[^\\\"]*\\\")|[^ ]+)" matches "${args}")
  string(REGEX REPLACE "(^\\\")|(\\\"$)" "" matches "${matches}")
  string(REGEX REPLACE "(;\\\")|(\\\";)" ";" matches "${matches}")
# hack for windows paths
  string(REPLACE "\\" "/" matches "${matches}")
  set("${result}" "${matches}" PARENT_SCOPE)
endfunction()