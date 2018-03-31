function ret=activation_out(x,type,deriv)
%type
%1->tanh
%2->ReLu
%3->linear
%deriv
%1->normal
%2->derived
if(deriv==1)%normal
    if(type==1)%tanh
        ret=2./(1+exp(-x))-1;
    elseif(type==2)%ReLu
        ret=log(1+exp(x));
    elseif(type==3)%linear
        ret=x;
    end
elseif(deriv==2)
    if(type==1)%tanh'
        ret=(1 + x) .* (1 - x);
    elseif(type==2)
        ret=1./(1+exp(-x));
    elseif(type==3)
        ret=ones(size(x));
    end
end
end