function [train,test] = loadSpam()
    X = importdata("spambase.data");
    y = X(:,end);
    y = y';
    X = X(:,1:end-1);
    X = X';
    I = randperm(length(y));
    y=y(I); % labels in range 1 to 10
    X=X(:,I);
    
    s=std(X,[],2);
    m=mean(X,2);
    X=bsxfun(@minus, X, m);
    X=bsxfun(@rdivide, X, s+.1);
    
    train.X = X(:,1:2300);
    train.y = y(1:2300);
    

    test.X = X(:,2302:end);
    test.y = y(2302:end);   
end