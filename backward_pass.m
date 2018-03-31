function [dely_delx,network]=backward_pass(network,delE_dely,X,maxormin)
%%%%% VERIFICATION PHASE %%%%%
% determine number of input samples, desired output and their dimensions
[P,N] = size(X);
L=network.structure;
%%%%% INITIALIZATION PHASE %%%%%
nLayers = length(L); % we'll use the number of layers often  
n=network.learning_rate;
m=.2;
% randomize the weight matrices (uniform random values in [-1 1], there
% is a weight matrix between each layer of nodes. Each layer (exclusive the 
% output layer) has a bias node whose activation is always 1, that is, the 
% node function is C(net) = 1. Furthermore, there is a link from each node
% in layer i to the bias node in layer j (the last row of each matrix)
% because it is less computationally expensive then the alternative. The 
% weights of all links to bias nodes are irrelevant and are defined as 0
w = network.weights;
%%%%% PREALLOCATION PHASE %%%%%
% for faster computation preallocate activation,net,prev_w and sum_w

% Activation: there is an activation matrix a{i} for each layer in the 
% network such that a{1} = the network input and a{end} = network output
% Since we're doing batch mode, each activation matrix a{i} is a 
% P-by-K (P=num of samples,K=nodes at layer i) matrix such that 
% a{i}(j) denotes the activation vector of layer i for the jth input and 
% a{i}(j,k) is the activation(output) of the kth node in layer i for the jth 
% input
a = cell(nLayers,1);  % one activation matrix for each layer
a{1} = [X ones(P,1)]; % a{1} is the input + '1' for the bias node activation
                      % a{1} remains the same throught the computation

for i=2:nLayers-1
    a{i} = ones(P,L(i)+1); % inner layers include a bias node (P-by-Nodes+1) 
end
a{end} = ones(P,L(end));   % no bias node at output layer

% Net: like activation, there is a net matrix net{i} for each layer
% exclusive the input such that net{i} = sum(w(i,j) * a(j)) for j = i-1
% and each net matrix net{i} is a P-by-K matrix such that net{i}(j) denotes 
% the net vector at layer i for the jth sample and net{i}(j,k) denotes the
% net input at node k of the ith layer for the jth sample
net = cell(nLayers-1,1); % one net matrix for each layer exclusive input
for i=1:nLayers-2;
    net{i} = ones(P,L(i+1)+1); % affix bias node 
end
net{end} = ones(P,L(end));

% Since we're using batch mode and momentum, two additional matrices are
% needed: prev_dw is the delta weight matrices at time (t-1) and sum_dw
% is the sum of the delta weights for each presentation of the input
% the notation here is the same as net and activation, that is prev_dw{i} 
% is P-by-K matrix where prev_dw{i} is the delta weight matrix for all samples 
% at time (t-1) and sum_dw{i} is a P-by-K matrix where sum_dw{i} is the
% the sum of the weight matrix at layer i for all samples
prev_dw = cell(nLayers-1,1);
sum_dw = cell(nLayers-1,1);
for i=1:nLayers-1
    prev_dw{i} = zeros(size(w{i})); % prev_dw starts at 0
    sum_dw{i} = zeros(size(w{i}));
end    
maxIter=1;
iter=1;
% loop until computational bounds are exceeded or the network has converged
% to a satisfactory condition. We allow for 30000 epochs here, it may be
% necessary to increase or decrease this bound depending on the number of 
% training
while iter<=maxIter
    iter=iter+1;
    % FEEDFORWARD PHASE: calculate input/output off each layer for all samples
    for i=1:nLayers-1
        net{i} = a{i} * w{i}'; % compute inputs to current layer
        % compute activation(output of current layer, for all layers
        % exclusive the output, the last node is the bias node and
        % its activation is 1
        if i < nLayers-1 % inner layers
            a{i+1}=[...
            activation_out(net{i}(:,1:end-1),network.act_type(i+1),1),...
            ones(P,1)];
        else             % output layers
            %a{i+1} = 2 ./ (1 + exp(-net{i})) - 1;
            a{i+1}=activation_out(net{i},network.act_type(i+1),1);
        end
    end
    
    % calculate sum squared error of all samples
    %err = (D-a{end});       % save this for later
    err=delE_dely;
    sse = sum(sum(err.^2));
    % BACKPROPAGATION PHASE: calculate the modified error at the output layer: 
    % S'(Output) * (D-Output) in this case S'(Output) = (1+Output)*(1-Output)
    % then starting at the output layer, calculate the sum of the weight 
    % matrices for all samples: LearningRate * ModifiedError * Activation
    % then backpropagate the error such that the modified error for this
    % layer is: S'(Activation) * ModifiedError * weight matrix
%     delta = err .* (1 + a{end}) .* (1 - a{end});
    dely_delsum=activation_out(a{end},network.act_type(end),2);
    delta=err.*dely_delsum;
    delta2=dely_delsum;
    for i=nLayers-1:-1:1
        sum_dw{i} = n * delta' * a{i};
        if i > 0
            %delta = (1+a{i}) .* (1-a{i}) .* (delta*w{i});
            dely_delsum=activation_out(a{i},network.act_type(i),2);
            delta = dely_delsum.*(delta*w{i});
            %delQ_dela formed at i==0 i think
            %delta2=(1+a{i}) .* (1-a{i}) .* (delta2*w{i});
            delta2 = dely_delsum.*(delta2*w{i});
        end
    end
    
    % update the prev_w, weight matrices, epoch count and mse
    for i=1:nLayers-1
        % we have the sum of the delta weights, divide through by the 
        % number of samples and add momentum * delta weight at (t-1)
        % finally, update the weight matrices
        prev_dw{i} = (sum_dw{i} ./ P) + (m * prev_dw{i});
        temp_prev=maxormin*prev_dw{i};
        w{i} = w{i} + temp_prev;%+ minimizes i think
    end 
end
% return the trained network
network.weights = w;
network.mse = sse/(P*network.structure(end));
dely_delx=delta2;

% if(any(isnan(w{3}(:)))==1 || any(isnan(w{2}(:)))==1 || any(isnan(w{1}(:)))==1)
% 	fprintf('*')
% end
end