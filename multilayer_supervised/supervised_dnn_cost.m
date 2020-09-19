function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1; 
hAct = cell(numHidden+2, 1); 
gradStack = cell(numHidden+1, 1); 
hAct{1} = data; % set first layer to inputs
m = size(data,2);
%% forward prop

% z(l+1) = W(l)a(l) + b(l)
% a(l+1) = f(z(l+1)
numLayers = numHidden + 1; 
for i=1:numLayers
   hAct{i+1} = stack{i}.W*hAct{i} + repmat(stack{i}.b,1,m);
   if i == numLayers
       final = hAct{i+1};
   end
   hAct{i+1} = sigmoid(hAct{i+1});
end

% final layer contains the predictions
pred_prob = hAct{numLayers+1};

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
% same as cost for softmax
exps = exp(final);
prob = bsxfun(@rdivide,exps,sum(exps));
actual = full(sparse(labels,1:m,1));
cost = -sum(sum(log(prob) .* actual));

%% compute gradients using backpropagation
epsilon = -(actual - prob); %output layer gradient
for i=numLayers:-1:1
    gradStack{i} = struct;
    gradStack{i}.W = epsilon*hAct{i}'/m; % divide each entry by the total
    gradStack{i}.b = sum(epsilon,2)/m; % sum across columns and divide by total
    epsilon = (stack{i}.W'*epsilon).*hAct{i}.*(1 - hAct{i});
end

[grad] = stack2params(gradStack);
end