## shorthand to fork a cmake script
function(process_start_script scriptish)
  fwrite_temp("${scriptish}" ".cmake")
  ans(script_path)
  execute(
    COMMAND
    "${CMAKE_COMMAND}"
    -P
    "${script_path}"
    ${ARGN}
    --async
  )
  return_ans()
endfunction()