##
## 
## creates a temporary file containing the specified content
## returns the path for that file 
function(fwrite_temp content)
    set(ext ${ARGN})

    if (NOT ext)
        set(ext ".txt")
    endif ()

    cmakepp_config(temp_dir)
    ans(temp_dir)

    path_vary("${temp_dir}/fwrite_temp${ext}")
    ans(temp_path)

    fwrite("${temp_path}" "${content}")

    return_ref(temp_path)
endfunction()
