## returns true iff cmake is currently in script mode
function(is_script_mode)
 commandline_get()
 ans(args)

 list_extract(args command P path)
 if("${P}" STREQUAL "-P")
  return(true)
else()
  return(false)
 endif()
endfunction()

## returns the file that was executed via script mode
function(script_mode_file)
  commandline_get()
  ans(args)

 list_extract(args command P path)
if(NOT "${P}" STREQUAL "-P")
  return()
endif()
  path("${path}")
  ans(path)
  return_ref(path)
endfunction()

