##
## `(<template_path:<file path>>)-><generated content:string>`
##  
## opens the specified template and runs it in its directory
## keeps track of recursive template calling
## * returns 
##    * the output of the template
## * scope
##    * `pwd()` is set to the templates path
##    * `${template_path}` is set to the path of the current template
##    * `${template_dir}` is set to the directory of the current template
##    * `${root_template_dir}` is set to the directory of the first template run
##    * `${root_template_path}` is set to the path of the first template run
##    * `${parent_template_dir}` is set to the calling templates dir 
##    * `${parent_template_path}`  is set to the calling templates path
## 
## 
function(template_run_file template_path)
    template_compile_file("${template_path}")
    ans(template)

    get_filename_component(template_dir "${template_path}" PATH)

    if (NOT root_template_path)
        set(root_template_path "${template_path}")
        get_filename_component(root_template_dir "${template_path}" PATH)
        set(parent_template_dir)
        set(parent_template_path)
    endif ()
    set(parent_template_path "${template_path}")
    set(parent_template_dir "${template_dir}")
    path_relative("${root_template_path}" "${template_path}")
    ans(relative_template_path)

    path_relative("${root_template_dir}" "${template_dir}")
    ans(relative_template_dir)

    pushd("${template_dir}")
    eval("${template}")
    ans(result)
    popd()
    return_ref(result)
endfunction()