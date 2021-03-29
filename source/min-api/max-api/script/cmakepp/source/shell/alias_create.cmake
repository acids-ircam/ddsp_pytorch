# creates a systemwide alias callend ${name} which executes the specified command_string
#  you have to restart you shell/re-login under windows for changes to take effect 
function(alias_create name command_string)


  if(WIN32)      
    cmakepp_config(bin_dir)
    ans(bin_dir)
    set(path "${bin_dir}/${name}.bat")
    fwrite("${path}" "@echo off\r\n${command_string} %*")
    reg_append_if_not_exists(HKCU/Environment Path "${bin_dir}")
    ans(res)
    if(res)
      #message(INFO "alias ${name} was created - it will be available as soon as you restart your shell")
    else()
      #message(INFO "alias ${name} as created - it is directly available for use")
    endif()
    return(true)
  endif()


  shell_get()
  ans(shell)

  if("${shell}" STREQUAL "bash")
    path("~/.bashrc")
    ans(bc)
    fappend("${bc}" "\nalias ${name}='${command_string}'")
    #message(INFO "alias ${name} was created - it will be available as soon as you restart your shell")

  else()
    message(FATAL_ERROR "creating alias is not supported by cmakepp on your system your current shell (${shell})")
  endif()
endfunction()

