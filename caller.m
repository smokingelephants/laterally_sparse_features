function caller()
addpath(genpath('../../../forkable_utils'))
% x = loadMNISTImages('train-images.idx3-ubyte');

[train_x,train_t,val_x,val_t]=gentempdata();
% nums=size(x,2);
% numtrain=ceil(.7*nums);
% % ids=randperm(nums);
% % x=images(:,ids);
% % for i=1:size(images,1)
% %     x(i,:)=x(i,:)-min(x(i,:));
% %     x(i,:)=x(i,:)./max(x(i,:));
% % end
% train_x=x(:,1:numtrain);
% val_x=x(:,numtrain+1:end);
disp('initializing complete... training started')
weights=nwbprop(train_x,val_x,10);


end