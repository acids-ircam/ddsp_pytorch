



   
    ## if `mapOrDefaultValue` is a map then just returns the map
    ## if not `mapOrDefaultValue` is assigned to a new map under the specified defaultKey
    function(map_coerce mapOrDefaultValue defaultKey) 
      is_map("${mapOrDefaultValue}")
      ans(isMap)
      if(NOT isMap)
        map_new()
        ans(map)
        map_set(${map} ${defaultKey} ${mapOrDefaultValue})
      else()
        set(map "${mapOrDefaultValue}")
      endif()
      return(${map})
    endfunction()