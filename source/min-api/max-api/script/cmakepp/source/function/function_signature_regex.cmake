function(function_signature_regex result)
	set(${result} "^[ ]*([mM][aA][cC][rR][oO]|[fF][uU][nN][cC][tT][iI][oO][nN])[ ]*\\([ ]*([A-Za-z0-9_\\\\-]*)(.*)\\)" PARENT_SCOPE)
endfunction()