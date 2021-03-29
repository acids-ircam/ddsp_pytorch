# swaps the element of lst at i with element at index j
macro(list_swap __list_swap_lst i j)
	list(GET ${__list_swap_lst} ${i} a)
	list(GET ${__list_swap_lst} ${j} b)
	list_replace_at(${__list_swap_lst} ${i} ${b})
	list_replace_at(${__list_swap_lst} ${j} ${a})
endmacro()

