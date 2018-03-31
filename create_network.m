function Network=create_network(L,ntype)
%ntype=1 :actor network
%ntype=2 :Q network
%%%%% INITIALIZATION PHASE %%%%%
nLayers = length(L); % we'll use the number of layers often  

% randomize the weight matrices (uniform random values in [-1 1], there
% is a weight matrix between each layer of nodes. Each layer (exclusive the 
% output layer) has a bias node whose activation is always 1, that is, the 
% node function is C(net) = 1. Furthermore, there is a link from each node
% in layer i to the bias node in layer j (the last row of each matrix)
% because it is less computationally expensive then the alternative. The 
% weights of all links to bias nodes are irrelevant and are defined as 0
if(ntype==1)
    mul=1;
elseif(ntype==2)
    mul=1;
end
    
w = cell(nLayers-1,1); % a weight matrix between each layer
for i=1:nLayers-2        
    w{i} = [1 - (2*mul).*rand(L(i+1),L(i)+1) ; zeros(1,L(i)+1)];
end
w{end} = 1 - (2*mul).*rand(L(end),L(end-1)+1);

Network.structure = L;
Network.weights = w;
Network.epochs = -1;
Network.mse = -1;

if(ntype==1)%actor
    Network.act_type=ones(1,nLayers);
    Network.act_type(1,1)=3; %input layer linear
    Network.learning_rate=1e-6;
elseif(ntype==2)%critic
    Network.act_type=1*ones(1,nLayers);
%     Network.act_type(1,[1,end])=3; %input layer linear
    Network.act_type(1,1)=3;
    Network.learning_rate=1e-3;
end



%Network.learning_rate=1e-6;
%1->tanh
%2->ReLu
%3->linear
end