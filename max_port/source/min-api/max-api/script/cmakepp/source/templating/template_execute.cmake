## `(<template file:<file path>> <?output file:<file path>>)-><file path>`
##
## compiles the specified template file to the speciefied output file
## if no output file is given the template file is expected to end with `.in`) and the 
## output file will be set to the same path without the `.in` ending
##
## Uses  see [`template_run_file`](#template_run_file) internally. 
##
## returns the path to which it was compiled
##
function(template_execute template_path)
    set(args ${ARGN})
    list_pop_front(args)
    ans(output_file)
    if (NOT output_file)
        if (NOT "${template_path}" MATCHES "\\.in$")
            message(FATAL_ERROR "expected a '.in' file")
        endif ()
        string(REGEX REPLACE "(.+)\\.in" "\\1" output_file "${template_path}")
    endif ()

    template_run_file("${template_path}")
    ans(generated_content)
    fwrite("${output_file}" "${generated_content}")

    return("${output_file}")
endfunction()
