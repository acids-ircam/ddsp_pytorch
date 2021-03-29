
  ## unpacks the specified reference to a map
  ## let a map be stored in the var 'themap'
  ## let it have the key/values a/1 b/2 c/3
  ## map_unpack(themap) will create the variables
  ## ${themap.a} contains 1
  ## ${themap.b} contains 2
  ## ${themap.c} contains 3
  function(map_unpack __ref)
    map_iterator(${${__ref}})
    ans(it)
    while(true)
      map_iterator_break(it)
      set("${__ref}.${it.key}" ${it.value} PARENT_SCOPE)
    endwhile()
  endfunction()