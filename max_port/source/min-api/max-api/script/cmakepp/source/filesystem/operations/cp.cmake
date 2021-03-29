# copies the specified path to the specified target
# if last argument is a existing directory all previous files will be copied there
# else only two arguments are allow source and target
# cp(<sourcefile> <targetfile>)
# cp([<sourcefile> ...] <existing targetdir>)
function(cp)
  set(args ${ARGN})
  list_pop_back(args)
  ans(target)

  list_length(args)
  ans(len)
  path("${target}")
  ans(target)
  # single move

  if(NOT IS_DIRECTORY "${target}" )
    if(NOT "${len}" EQUAL "1")
      message(FATAL_ERROR "wrong usage for cp() exactly one source file needs to be specified")
    endif() 
    path("${args}")
    ans(source)
    # this just has to be terribly slow... 
    # i am missing a direct
    cmake_lean(-E "copy" "${source}" "${target}")
    ans_extract(error)
    if(error)
      message("failed to copy ${source} to ${target}")
    endif()
   return()
  endif()


  paths(${args})
  ans(paths)
  file(COPY ${paths} DESTINATION "${target}") 
  

  return()
endfunction()

