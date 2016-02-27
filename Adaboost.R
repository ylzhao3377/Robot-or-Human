Adaboost = function(weak_cf_train,outcome){  
  myalpha = NULL
  error = NULL
  Z=NULL
  w = 1/nrow(weak_cf_train)
  for(i in 1:ncol(weak_cf_train)){
    e = sum((weak_cf_train[,i] != outcome)*w)
    alpha = 0.5*log2((1-e)/e)    
    z = sum(w*exp(-alpha*(weak_cf_train == outcome)))
    w = w*exp(-alpha*(weak_cf_train[,i] == outcome))/z
    myalpha = c(myalpha,alpha)
    error = c(error,e)
    Z = c(Z,z)
  }
  myalpha = as.matrix(myalpha)
  return(myalpha)  
}


