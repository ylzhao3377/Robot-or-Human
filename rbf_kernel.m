function [K] = rbf_kernel(x,y,gamma)
    for i = 1:size(x,1)
        for j = 1:size(y,1)
           K(i,j) = exp(-gamma*norm(x(i,:)-y(j,:))^2);
        end
    end
end