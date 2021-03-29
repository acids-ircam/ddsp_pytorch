## `()-><qualified path>`
##
## returns the current users home directory on all OSs
## 
function(home_dir)
    shell_get()
    ans(shell)
    if ("${shell}" STREQUAL "cmd")
        shell_env_get("HOMEDRIVE")
        ans(dr)
        shell_env_get("HOMEPATH")
        ans(p)
        set(res "${dr}${p}")
        file(TO_CMAKE_PATH "${res}" res)
        path("${res}")
        ans(res)
    elseif ("${shell}" STREQUAL "bash")
        shell_env_get(HOME)
        ans(res)
    else ()
        message(FATAL_ERROR "supported shells: cmd & bash")
    endif ()
    eval("
    function(home_dir)
      set(__ans \"${res}\" PARENT_SCOPE)
    endfunction()
      ")
    return_ref(res)
endfunction()
