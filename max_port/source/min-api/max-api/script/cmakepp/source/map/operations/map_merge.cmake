# creates a union from all all maps passed as ARGN and combines them in result
# you can merge two maps by typing map_union(${map1} ${map1} ${map2})
# maps are merged in order ( the last one takes precedence)
function(map_merge )
	set(lst ${ARGN})

	map_new()
  ans(res)
  
	foreach(map ${lst})
		map_keys(${map} )
		ans(keys)
		foreach(key ${keys})
			map_tryget(${res}  ${key})
			ans(existing_val)
			map_tryget(${map}  ${key})
			ans(val)

			is_map("${existing_val}" )
			ans(existing_ismap)
			is_map("${val}" )
			ans(new_ismap)

			if(new_ismap AND existing_ismap)
				map_union(${existing_val}  ${val})
				ans(existing_val)
			else()
				
				map_set(${res} ${key} ${val})
			endif()
		endforeach()
	endforeach()
	return(${res})
endfunction()

