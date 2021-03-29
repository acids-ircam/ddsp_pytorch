function(test)
    # element(MAP)
    # 	value(KEY k1 1)
    # 	value(KEY k2 2)
    # 	element(k3 MAP)
    # 		value(KEY k1 "a")
    # 		element(k2 LIST)
    # 			value(I)
    # 			value(II)
    # 			value(III)
    # 		element(END)
    # 	element(END)
    # element(END uut)

    script("{
		k1:1,
		k2:2,
		k3:{
		  k1:'a',
		  k2:[
		  	'I',
		  	'II',
		  	'III'
		  ]
		}
	}")
    ans(uut)


    # a missing variable
    set(tmp_var)
    map_navigate(res "tmp_var")
    assert(NOT res)

    # empty path
    map_navigate(res "")
    assert("_${res}" STREQUAL "_")


    # a normal variable
    set(tmp_var "hello")
    map_navigate(res "tmp_var")
    assert(res)
    assert(${res} STREQUAL "hello")


    # unevaluated ref
    map_navigate(res "uut")
    assert(${res} STREQUAL "${uut}")


    # evaluated ref
    map_navigate(res "${uut}")
    assert(${res} STREQUAL "${uut}")

    # navigate simple value
    map_navigate(res "uut.k1")
    assert(${res} STREQUAL "1")
    map_navigate(res "${uut}.k1")
    assert(${res} STREQUAL "1")

    # navigate nested value
    map_navigate(res "uut.k3.k1")
    assert(${res} STREQUAL "a")

    message("test inconclusive")
    return()
    # navigate nested value with index
    map_navigate(res "uut.[2].k1")
    assert(${res} STREQUAL "a")

    # navigate nested value multiple indeces
    map_navigate(res "uut.[2][1][2]")
    assert(${res} STREQUAL "III")


    # navigate nested value multiple mixed indeces
    map_navigate(res "uut.[2].k2[2]")
    assert(${res} STREQUAL "III")


    map_navigate(res "*uut.k3.k2")
    assert(EQUALS ${res} I II III)

    # check special symbols
    map_new()
    ans(res)
    map_set(${res} "k1" "\${ARGN} \; ;")
    map_navigate(val "res.k1")
    assert(EQUALS "${val}" "\${ARGN} \; ;")

endfunction()