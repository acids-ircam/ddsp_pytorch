
## executes a cmake script in configure mode
## parameters to script will be passed to cmake 

parameter_definition(
  cmake_configure_script 
  <--script:<string>>
  [--target-dir:<path>]
)
function(cmake_configure_script)
  arguments_extract_defined_values(0 ${ARGC} cmake_configure_script)
  ans(args)

  if(target_dir)
    pushd("${target_dir}" --create)
    ans(dir)
  else()
    pushtmp()
    ans(dir)
  endif()
  


  log("executing cmake configure script in '${dir}'")

  cmakepp_config(cmakepp_path)
  ans(cmakepp_path)

  path("output.qm")
  ans(output_file)

  set(cmakelists_content "
    cmake_minimum_required(VERSION 2.8.12)
    include(${cmakepp_path})
   
    set_ans()
    function(___execute_it)
      ${script}
      return_ans()
    endfunction()
    ___execute_it()
    ans(result)        
    cmake_write(\"${output_file}\" \"\${result}\")
  ")

  fwrite("CMakeLists.txt" "${cmakelists_content}") 


  pushd(build --create)
    cmake_lean(".." ${args})
    ans_extract(error)
    ans(stdout)
  popd()
  
  

  if(error)      
    error("error out: ${stdout}")
    set(res)
  else()
    cmake_read(${output_file})
    ans(res)  
  endif()

  if(passthru)
    message("${stdout}")
  endif()

  if(target_dir)
    popd()
  else()
    poptmp()  
  endif()


  return_ref(res)
endfunction()


