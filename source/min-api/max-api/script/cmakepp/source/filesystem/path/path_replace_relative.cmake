
# replaces all fully qualified paths with a path relative to 
# specified path (using .. to navigate upward)
function(path_replace_relative path)
    set(args ${ARGN})

    set(parentDirs ".")
    path("${path}")
    ans(current_path)
    
    while(true)            
        string(REPLACE "${current_path}" "${parentDirs}" args ${args})
        path_parent_dir("${current_path}")
        ans(next_path)

        if("${current_path}_" STREQUAL "${next_path}_")
            return_ref(args)
        endif()

        set(current_path "${next_path}")


        if("${parentDirs}" STREQUAL ".")
            set(parentDirs "..")
        else()
            set(parentDirs "${parentDirs}/..")
        endif()
    endwhile()
    
endfunction()