function y_hat = svm_rbf_poly(X,group,c,y,kfun,para)
n = size(X,1);
K = feval(kfun,X,X,para);
l = eps^.5;
H = (group*group').*K + l*eye(n);
f = -ones(n,1);
A = zeros(1,n);
a = 0;
Aeq = group';
beq = 0;
lb = zeros(n,1);
ub = c * ones(n,1);
lambda0 = ones(n,1);
group(group == 0) = -1;
options = optimoptions('quadprog','Algorithm','interior-point-convex',...
    'Display','off');
alpha = quadprog(H,f,[],[],Aeq,beq,lb,ub,lambda0,options);
font = sum(repmat(alpha.*group,1,n).*K,1)';
svIndex = find(alpha > sqrt(eps));
sv = X(svIndex,:);
l_hat = group(svIndex).*alpha(svIndex);
%Calculate the estimator
[~,maxPos] = max(alpha);
b_hat = mean(group(svIndex)-font(svIndex));
output_vali = (feval(kfun,sv,y,para)'*l_hat(:)) + b_hat;
y_hat = sign(output_vali);
y_hat(y_hat == -1) = 0;
%Fit the training set
output_train = (feval(kfun,sv,X,para)'*l_hat(:)) + b_hat;
y_train = sign(output_train);
er_train = 1 - sum(y_train == group)/n;
end