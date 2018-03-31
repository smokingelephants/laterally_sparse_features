function out=forward_pass(network,input)
w=network.weights;
L=network.structure;
X=input;
nLayers=length(L);
[P,N] = size(X);

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

    % FEEDFORWARD PHASE: calculate input/output off each layer for all samples
for i=1:nLayers-1
    net{i} = a{i} * w{i}'; % compute inputs to current layer
    % compute activation(output of current layer, for all layers
    % exclusive the output, the last node is the bias node and
    % its activation is 1
    if i < nLayers-1 % inner layers
       % a{i+1} = [2./(1+exp(-))-1 ones(P,1)];
        a{i+1}=[...
            activation_out(net{i}(:,1:end-1),network.act_type(i+1),1),...
            ones(P,1)];
    else             % output layers
        %a{i+1} = 2 ./ (1 + exp(-net{i})) - 1;
        a{i+1}=activation_out(net{i},network.act_type(i+1),1);
    end
end
out=a{end};
end