function [train_x,train_t,val_x,val_t]=gentempdata()
numi=4;
numo=1;
nums=1500;
train_x=rand(numi,nums);
train_t=rand(numo,nums);
val_x=rand(numi,nums/2);
val_t=rand(numo,nums/2);
end