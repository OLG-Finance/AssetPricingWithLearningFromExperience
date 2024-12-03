function [Deltabar,IntVec,Xt,Delta_s_t,Yt,Zt,f,tau] = BuildUpCohortsMAIN(dZt,Nt,dt,rho,nu,Vbar,mu_Y,sigma_Y,bet,That)
% Builds up a sufficiently large set of cohorts
Npre = That/dt;
Zt = [0;cumsum(dZt)];
yg = (mu_Y-0.5*sigma_Y^2)*dt*ones(Nt-1,1)+sigma_Y*dZt;
Yt = [1;exp(cumsum(yg))]; 
Xt = ones(Nt,1)*nu*bet; 
Deltabar = zeros(Nt,1);
Delta_s_t = 0;
IntVec = 1*nu*bet;
tau = dt;
reduction = exp(-nu*dt);
for i=2:Nt
     Part = IntVec.*exp(-(rho+0.5*Delta_s_t.*Delta_s_t)*dt + Delta_s_t*dZt(i-1));
     if i==2
        Xt(i) = Part;
        Deltabar(i) = Part*Delta_s_t;
     else
        Xt(i) = sum(Part);
        Deltabar(i) = sum(Part.*Delta_s_t)/Xt(i); 
     end
     IntVec = [reduction*Part bet*(1-reduction)*Xt(i)];
     f = IntVec./Xt(i);   
     dDelta_s_t = (PostVar(sigma_Y,Vbar,tau)/sigma_Y^2).*(-Delta_s_t*dt+ones(size(Delta_s_t))*dZt(i-1));
     if i<(Npre+1)
        Delta_s_t = [Delta_s_t+dDelta_s_t 0];
     else
        DELbias = sum(dZt(i-Npre:i-1))/That;
        Delta_s_t = [Delta_s_t+dDelta_s_t DELbias];
     end     
     tau=[tau+dt 0];
end

end
