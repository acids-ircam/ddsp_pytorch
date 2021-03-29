# runs a shell script on the current platform
# not that
function(shell cmd)

    shell_get()
    ans(shell)
    if ("${shell}" STREQUAL "cmd")
        fwrite_temp("@echo off\n${cmd}" ".bat")
        ans(shell_script)
    elseif ("${shell}" STREQUAL "bash")
        fwrite_temp("#!/bin/bash\n${cmd}" ".sh")
        ans(shell_script)
        # make script executable
        execute_process(COMMAND "chmod" "+x" "${shell_script}")
    else ()
        message(FATAL_ERROR "Shell not suported: ${shell}")
    endif ()

    # execute shell script which write the keyboard input to the ${value_file}
    set(args ${ARGN})

    list_extract_flag(args --process-handle)
    ans(return_process_handle)

    execute("${shell_script}" ${args} --process-handle)
    ans(res)

    # remove temp file
    file(REMOVE "${shell_script}")
    if (return_process_handle)
        return_ref(res)
    endif ()

    map_tryget(${res} exit_code)
    ans(exit_code)

    if (NOT "_${exit_code}" STREQUAL "_0")
        message(ERROR "Shell failed with exit code ${exit_code}")
        return()
    endif ()

    map_tryget(${res} stdout)
    ans(stdout)
    return_ref(stdout)
endfunction()