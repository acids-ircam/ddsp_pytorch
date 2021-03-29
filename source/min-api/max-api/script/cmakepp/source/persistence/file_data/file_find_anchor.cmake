## `(<search file:<file name>> [<location:<path>>])-><path>|<null>`
##
## an anchor file is what I call a file that exists somewhere in the
## specified location, any parent directory, or in the current directory
## for example git normally uses an anchorfile in every repository
## (in that cast the `.git` folder)
## also alot of projects use a local file system and in the project;'s
## root folder there exists an anchor file e.g. `.cps` `.cps/project.scmake` 
##
function(file_find_anchor search_file)
  set(search ${ARGN})
  path("${search}")
  ans(search)
  set(current_path "${search}")
  set(last_path)
  while(true)
    if("${last_path}_" STREQUAL "${current_path}_")
      return()
    endif()
    set(anchor_file "${current_path}/${search_file}")
    if(EXISTS "${anchor_file}")
      break()
    endif()
    set(last_path "${current_path}")
    path_parent_dir("${current_path}")
    ans(current_path)
  endwhile()
  return_ref(anchor_file)
endfunction()
