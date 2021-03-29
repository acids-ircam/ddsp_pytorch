# comapres two lists with each other
# usage
# list_equal( 1 2 3 4 1 2 3 4)
# list_equal( listA listB)
# list_equal( ${listA} ${listB})
# ...
# COMPARATOR defaults to STREQUAL
# COMPARATOR can also be a lambda expression
# COMPARATOR can also be EQUAL
function(list_equal)
	set(options)
  	set(oneValueArgs COMPARATOR)
  	set(multiValueArgs)
  	set(prefix)
  	cmake_parse_arguments("${prefix}" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
	#_UNPARSED_ARGUMENTS


	# get length of both lists

	list(LENGTH _UNPARSED_ARGUMENTS count)



	#if count is exactly two input could be list references
	if(${count} EQUAL 2)
		list(GET _UNPARSED_ARGUMENTS 0 ____listA)
		list(GET _UNPARSED_ARGUMENTS 1 ____listB)
		if(DEFINED ${____listA} AND DEFINED ${____listB})
			# recursive call and return
			list_equal(  ${${____listA}} ${${____listB}} COMPARATOR "${_COMPARATOR}")
			return_ans()
		endif()

	endif()

	set(listA)
	set(listB)




	math(EXPR single_count "${count} / 2")
	math(EXPR is_even "${count} % 2")
	if(NOT ${is_even} EQUAL "0")
		#element count is not divisible by two so the lists cannot be equal
		# because they do not have the same length

		return(false)

	else()
		# split input arguments into two
		list_split(listA listB _UNPARSED_ARGUMENTS ${single_count})
	#message("${_UNPARSED_ARGUMENTS} => ${listA} AND ${listB}")
	endif()


	# set default comparator to strequal
	if(NOT _COMPARATOR)
		set(_COMPARATOR "STREQUAL")
	endif()

	# depending on the comparator
	if(${_COMPARATOR} STREQUAL "STREQUAL")
		set(lambda "[](a b) eval_truth('{{a}}' STREQUAL '{{b}}')")
	elseif(${_COMPARATOR} STREQUAL "EQUAL")
		set(lambda "[](a b) eval_truth('{{a}}' EQUAL '{{b}}')")
	else()
		set(lambda "${_COMPARATOR}")
	endif()
	# import function string 
	function_import("${lambda}" as __list_equal_comparator REDEFINE)
		
	set(res)
	# compare list
	math(EXPR single_count "${single_count} - 1")
	foreach(i RANGE ${single_count})
		list(GET listA ${i} a)
		list(GET listB ${i} b)
		#message("comparing ${a} ${b}")
		__list_equal_comparator(${a} ${b})
		ans(res)
		if(NOT res)
			return(false)
		endif()
	endforeach()
	return(true)

endfunction()