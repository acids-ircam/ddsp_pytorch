## returns the current error output
## This can change until the process is finished
function(process_stderr handle)
    process_handle("${handle}")
    ans(handle)
    map_tryget("${handle}" stderr_file)
    ans(stderr_file)
    fread("${stderr_file}")
    ans(stderr)
    return_ref(stderr)
endfunction()

