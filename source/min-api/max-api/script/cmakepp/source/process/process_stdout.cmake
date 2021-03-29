## returns the current stdout of a <process handle>
## this changes until the process is ove
function(process_stdout handle)
    process_handle("${handle}")
    ans(handle)
    map_tryget("${handle}" stdout_file)
    ans(stdout_file)
    fread("${stdout_file}")
    ans(stdout)
return_ref(stdout)
endfunction()
