function(test)
    home_dir()
    ans(home)

    path_qualify_from("/dir1" "")
    ans(res)
    assert("${res}" MATCHES "/dir1$")

    path_qualify_from("/dir1" "~")
    ans(res)
    assert("${res}" STREQUAL "${home}")

    path_qualify_from("/dir1" "~/")
    ans(res)
    assert("${res}" STREQUAL "${home}")

    path_qualify_from("/dir1" "/asd/asd")
    ans(res)
    assert("${res}" MATCHES "/asd/asd$")

    path_qualify_from("/dir1" "asd/asd")
    ans(res)
    assert("${res}" MATCHES "/dir1/asd/asd$")

    path_qualify_from("/dir1" "~/asd/asd")
    ans(res)
    assert("${res}" STREQUAL "${home}/asd/asd")
endfunction()