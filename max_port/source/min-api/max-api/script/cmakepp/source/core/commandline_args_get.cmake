## commandline_args_get([--no-script])-> <string...>
## 
## returns the command line arguments with which cmake 
## was without the executable
##
## --no-script flag removes the script file from the command line args
##
## Example:
## command line: 'cmake -P myscript.cmake a s d'
## commandline_args_get() -> -P;myscript;a;s;d
## commandline_args_get(--no-script) -> a;s;d

function(commandline_args_get)
  set(args ${ARGN})
  list_extract_flag(args --no-script)
  ans(no_script)
  commandline_get()
  ans(args)
  # remove executable
  list_pop_front(args)
  if(no_script)
    list_extract_labelled_value(args -P)
  endif()
  return_ref(args)
endfunction()