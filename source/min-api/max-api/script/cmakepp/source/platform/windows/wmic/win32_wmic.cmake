## wraps the windows wmic command (windows XP and higher )
# since wmic does outputs unicode and does not take forward slash paths the usage is more complicated 
# and wrap_executable does not work
function(win32_wmic)
  pwd()
  ans(pwd)
  fwrite_temp("")
  ans(tmp)
  file(TO_NATIVE_PATH "${tmp}" out)

  execute_process(COMMAND wmic /output:${out} ${ARGN} RESULT_VARIABLE res WORKING_DIRECTORY "${pwd}")  
  if(NOT "${res}" EQUAL 0 )
    return()
  endif()

  fread_unicode16("${tmp}")        
  return_ans()
endfunction()
