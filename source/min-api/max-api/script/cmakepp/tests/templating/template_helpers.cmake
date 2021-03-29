function(test)
    markdown_section("expr_example" "Example")
    ans(res)

#    message("${res}")
    assert("${res}" STREQUAL "## <a name=\"expr_example\"></a> Example")
endfunction()
