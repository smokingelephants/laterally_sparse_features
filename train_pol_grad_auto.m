function [netQ,addval,delQ_delA]=train_pol_grad_auto(netQ,records,nums,numa,damp)
list_rewards=records(:,(nums+numa+1));
inp_s=records(:,1:nums);
out_a=records(:,(1+nums):(nums+numa));
inp_ns=records(:,(nums+numa+2):((2*nums)+numa+1));
out_na=records(:,((2*nums)+numa+2):end);

Qout_s=forward_pass(netQ,[inp_s,out_a]);
Qout_ns=forward_pass(netQ,[inp_ns,out_na]);
Qout_ns=zeros(size(Qout_ns));
[delE_dely,addval]=computeQgradient(Qout_s,list_rewards,Qout_ns);

[delQ_delXA,netQ]=backward_pass(netQ,delE_dely,...
    [inp_s,out_a],1);
delQ_delA=damp.*delQ_delXA(:,nums+1:nums+numa);
end

function [delE_dely,s_addval]=computeQgradient(Q_out_s,rew,Q_out_ns)
alpha=.2;
gamma=.0;
rew=repmat(rew,1,size(Q_out_ns,2));
% target_Q_s=Q_out_s+(alpha.*(rew+(gamma.*Q_out_ns)-Q_out_s));
target_Q_s=rew+(gamma.*Q_out_ns);

delE_dely=target_Q_s-Q_out_s;
addval=(alpha.*(rew+(gamma.*Q_out_ns)-Q_out_s));
s_addval=sum(abs(addval(:)));
end