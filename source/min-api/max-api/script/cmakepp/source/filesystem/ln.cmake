## `(<target:<path>> <link:<path>>?)-><bool>` 
##
## creates a symlink from `<link>` to `<target>` on all operating systems
## (Windows requires NTFS filesystem)
## if `<link>` is omitted then the link will be created in the local directory 
## with the same name as the target
##
function(ln)
  wrap_platform_specific_function(ln)
  ln(${ARGN})
  return_ans()
endfunction()



function(ln_Linux target)
   set(args ${ARGN})

  path_qualify(target)

  list_pop_front(args)
  ans(link)
  if("${link}_" STREQUAL "_")
    get_filename_component(link "${target}" NAME )
  endif()

  path_qualify(link)
  execute_process(COMMAND ln -s "${target}" "${link}" RESULT_VARIABLE error ERROR_VARIABLE stderr)
  if(error)
    return(false)
  endif() 
  return(true)
endfunction()


function(ln_Windows target)
  set(args ${ARGN})

  path_qualify(link)

  list_pop_front(args)
  ans(link)

  if("${link}_" STREQUAL "_")
    get_filename_component(link "${target}" NAME )
  endif()

  path_qualify(target)


  if(EXISTS "${target}" AND NOT IS_DIRECTORY "${target}")
    set(flags "/H")
  else()
    set(flags "/D" "/J")
  endif()
  string(REPLACE "/" "\\" link "${link}")
  string(REPLACE "/" "\\" target "${target}")

 # print_vars(link target flags)
  win32_cmd_lean("/C" "mklink" ${flags} "${link}" "${target}")
  ans_extract(error)
  if(error)
    return(false)
  endif()
  return(true)
endfunction()


