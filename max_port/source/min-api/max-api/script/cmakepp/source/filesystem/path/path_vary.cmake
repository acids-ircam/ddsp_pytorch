## `(<path>)-><qualified path>`
##
## varies the specified path until it does not exist
## this is done  by inserting a random string into the path and doing so until 
## a path is vound whic does not exist
function(path_vary path)
  path_qualify(path)
  get_filename_component(ext "${path}" EXT)
  get_filename_component(name "${path}" NAME_WE)
  get_filename_component(base "${path}" PATH)
  set(rnd)
  while(true)
    set(path "${base}/${name}${rnd}${ext}")
    
    if(NOT EXISTS "${path}")
      return("${path}")
    endif()


    ## alternatively count up
    string(RANDOM rnd)
    set(rnd "_${rnd}")

  endwhile()
endfunction()