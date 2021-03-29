## assert allows assertion

# splits a list into two parts after the specified index
# example:
# set(lst 1 2 3 4 5 6 7)
# list_split(p1 p2 lst 3)
# p1 will countain 1 2 3
# p2 will contain 4 5 6 7
function(list_split part1 part2 _lst index)
	list(LENGTH ${_lst} count)
	#message("${count} ${${_lst}}")
	# subtract one because range goes to index count and should only got to count -1
	math(EXPR count "${count} -1")
	set(p1)
	set(p2)
	foreach(i RANGE ${count})
		#message("${i}")
		list(GET ${_lst} ${i} val)
		if(${i} LESS ${index} )
			list(APPEND p1 ${val})
		else()
			list(APPEND p2 ${val})
		endif()
	endforeach()
	set(${part1} ${p1} PARENT_SCOPE)
	set(${part2} ${p2} PARENT_SCOPE)
	return()
endfunction()


