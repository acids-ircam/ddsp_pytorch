## returns a <process handle> to a process that runs for n seconds
#todo create shims
function(process_timeout n)
    execute(${CMAKE_COMMAND} -E sleep ${n} --async)
    return_ans()
endfunction()