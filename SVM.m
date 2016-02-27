% Import data
feature = readtable('features.csv');

% Find non-numeric features
colname = feature.Properties.VariableNames;
for i = [1:width(feature)]
    v_is_cell(i) = iscell(feature.(colname{i}));
end
colname{(v_is_cell==1)}

% Conver to logical feature
feature.payment_account_prefix_same_as_address_prefix = ...
    categorical(feature.payment_account_prefix_same_as_address_prefix);
categories(feature.payment_account_prefix_same_as_address_prefix)
feature.payment_account_prefix_same_as_address_prefix = ...
    renamecats(feature.payment_account_prefix_same_as_address_prefix, {'0','1'});
feature.payment_account_prefix_same_as_address_prefix = ...
    double(feature.payment_account_prefix_same_as_address_prefix);

feature.sleep = categorical(feature.sleep);
categories(feature.sleep)
feature.sleep = renamecats(feature.sleep, {'0','1'});
feature.sleep = double(feature.sleep);

feature.most_common_country = categorical(feature.most_common_country);
categories(feature.most_common_country)
feature.most_common_country = renamecats(feature.most_common_country, ...
    cellstr(num2str((0:93)')));
feature.most_common_country = double(feature.most_common_country);

%% Find missing value
TF = ismissing(feature,{'' 'NA' NaN Inf});
MissN = sum(TF);
unique(sum(TF))
% Replace NaN with -9999
V = colname(MissN>0);
f2 = feature{:,V};
f2(isnan(f2)) = -9999;
f2(isinf(f2)) = 9999;
feature{:,V} = f2;

%% Feature selection and seperate train and test
% PCA (96% explained by first 5 features)
%attr = colname;
%attr(8) = [];
%attr = attr(3:end);
%y = feature{feature{:,'outcome'} == -1,attr};
%X = feature{feature{:,'outcome'} ~= -1,attr};
%group = feature{feature{:,'outcome'} ~= -1,'outcome'};
%[wcoeff,~,latent,~,explained] = pca(X);
%X = X*wcoeff;
%X = X(:,1:5);
%y = y*wcoeff;
%y(:,6:end) = [];

% Eigenvalues
%tmp = cov(feature{:,3:end});
%eigen = eig(tmp);

attr=colname([182 183]);
y = feature{feature{:,'outcome'} == -1,attr};
X = feature{feature{:,'outcome'} ~= -1,attr};
group = feature{feature{:,'outcome'} ~= -1,'outcome'};
bid_id = feature{feature{:,'outcome'} == -1,'bidder_id'};

% Cross validation
pos = find(group == 1);
neg = find(group == 0);
%% Grid search
C = 10.^(-2:5);             % Modify C set and G set to get the best grid
G = 10.^(-6:1);
for i = 1:size(C,2)
    for j = 1:size(G,2)
        test = [1:size(X,1)]';
        pos_train = datasample(pos,100,'Replace',false);
        neg_train = datasample(neg,500,'Replace',false);
        train = [pos_train; neg_train];
        test(train) = [];
        y_grid = svm_rbf_poly(X(train,:),group(train,:),C(i),...
            X(test,:),@rbf_kernel,G(j));
        er_grid(i,j)= 1 - sum(y_grid == group(test,:))/size(test,1);
    end
end

%% SVM Cross validation
k = 50;
for i = 1:k
    pos_train = datasample(pos,100,'Replace',false);
    neg_train = datasample(neg,1000,'Replace',false);
    train = [pos_train; neg_train];
    y_hat(:,i) = svm_rbf_poly(X,group,46,y,@rbf_kernel,0.0001);
end
% output = cellstr(num2str(mode(y_hat,2)));
y_hat = svm_rbf_poly(X,group,46,y,@rbf_kernel,0.0001);
output = cellstr(num2str(y_hat));
submission=cell2table([bid_id output],'VariableNames',{'bidder_id' 'prediction'});
writetable(submission,'submission.csv','Delimiter',',');








