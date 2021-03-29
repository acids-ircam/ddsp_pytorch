# use this macro inside of a while(true) loop it breaks when the iterator is over
# e.g. this prints all key values in the map
# while(true) 
#   map_iterator_break(myiterator)
#   message("${myiterator.key} = ${myiterator.value}")
# endwhile()
macro(map_iterator_break it_ref)
    map_iterator_next(${it_ref})
    if (${it_ref}.end)
        break()
    endif ()
endmacro()