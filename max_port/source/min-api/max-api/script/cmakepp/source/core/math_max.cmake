# retruns the larger of the two values
function(math_max a b)
  if(${a} GREATER ${b})
    return(${a})
  else()
    return(${b})
  endif() 
endfunction()