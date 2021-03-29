function(test)
    set(var1 abc)
    set(var2 cde)
    unset(var3)

#    string_eval("@var1@ \${var2} @var3@ \${var3}")
    string_eval("\${var2} \${var3}")
    ans(res)

    assert("${res}" STREQUAL "cde ")
endfunction()