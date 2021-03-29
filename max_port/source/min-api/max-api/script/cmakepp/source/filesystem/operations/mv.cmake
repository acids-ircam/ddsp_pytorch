# moves the specified path to the specified target
# if last argument is a existing directory all previous files will be moved there
# else only two arguments are allow source and target
# mv(<sourcefile> <targetfile>)
# mv([<sourcefile> ...] <existing targetdir>)
function(mv)
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
      message(FATAL_ERROR "wrong usage for mv() exactly one source file needs to be specified")
    endif()
    path("${args}")
    ans(source)
    file(RENAME "${source}" "${target}")
    return()
  endif()

  foreach(source ${args})
    path_file_name("${source}")
    ans(fn)
    mv("${source}" "${target}/${fn}")
  endforeach()

  return()
endfunction()
