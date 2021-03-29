
function(math_min a b)
  if(${a} LESS ${b})
    return(${a})
  else()
    return(${b})
  endif() 
endfunction()