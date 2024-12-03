function [Xt2,Deltabar2,part1,mu_S,mu_S_t,muhat_S_t,r_t,theta_t,Port, muC_s_t,sigmaC_s_t,BIGf,BIGDELTA,Et,Vt,dR] = SimCohortsMAIN(biasvec,dZt,Nt,tau,IntVec,Delta_s_t,dt,rho,nu,Vbar,mu_Y,sigma_Y,sigma_S,bet,That,Npre)

% Initializing variables

Xt2 = ones(Nt,1);
Deltabar2 = ones(Nt,1);
Et = ones(Nt,1);
Vt = ones(Nt,1);

part1=zeros(Nt,1);
dR = zeros(Nt,1);
counter = 1;
reduction = exp(-nu*dt);
BIGDELTA = zeros(Nt,Nt);
BIGf = zeros(Nt,Nt);

RevNt = Nt:-1:1;
fhat = reduction.^(RevNt-1)*nu*dt;
fhat = fhat./sum(fhat); 

% Expected Returns
mu_S=zeros(Nt,1);   %This is the expected return under true measure
mu_S_t=zeros(Nt,1); %This is the expected return under agent-measure
muhat_S_t=zeros(Nt,1); 

muC_s_t = zeros(Nt,1);      % Drift log consumption
sigmaC_s_t = zeros(Nt,1);   % Diffusion log consumption

% Aggregate quantities 
r_t=zeros(Nt,1);
theta_t=zeros(Nt,1);
fst=zeros(Nt,1);       % Individual consumption-wealth ratio
for i=1:Nt
    Part = IntVec.*exp(-(0.5*Delta_s_t.*Delta_s_t)*dt + Delta_s_t*dZt(i));
    if counter<Nt
        if counter==1
            part1(counter) =  sum(biasvec)/That;
        else
            part1(counter) = part1(counter-1)  +(PostVar(sigma_Y,Vbar,(counter*dt))/sigma_Y^2)*(-part1(counter-1)*dt+dZt(i-1));
        end
    end
    
    if i==1
        dR(i) = 0;
    else
        dR(i) = (mu_S(counter-1)-r_t(counter-1) + rho + mu_Y -sigma_Y^2+nu*(1-bet))*dt +sigma_S*dZt(i);
    end
    
    Xt2(counter) = sum(Part);
    Deltabar2(counter) = sum(Part.*Delta_s_t)/Xt2(counter); 
    f = Part./Xt2(counter);    
     

    
    BIGDELTA(i,:) = Delta_s_t;
  
    
    
    mu_S(counter)= sigma_S*sigma_Y-(sigma_S-sigma_Y)*Deltabar2(counter);
    mu_S_t(counter)=sigma_S*sigma_Y+sigma_S*(-Deltabar2(counter)+part1(counter))+sigma_Y*Deltabar2(counter); 
    muhat_S_t(counter)= mu_S(counter) + sigma_S*fhat*Delta_s_t';
   
    
   
    r_t(counter) = rho + mu_Y -sigma_Y^2+nu*(1-bet) +sigma_Y*Deltabar2(counter);
    theta_t(counter) = sigma_Y - Deltabar2(counter);
   
    
    
    Et(counter) = f*(Vbar./(1+(Vbar/sigma_Y^2)*dt*RevNt))'*(1/sigma_Y);
    Vt(counter) = (f*(Delta_s_t').^2 - Deltabar2(counter)^2)*sigma_Y;
    
    fst(i) = f(Nt-(i-1));
    BIGf(i,:) = f;
    
   

    muC_s_t(counter) = mu_Y + nu*(1-bet)+ (sigma_Y-Deltabar2(counter))*(part1(counter)-Deltabar2(counter));
    sigmaC_s_t(counter) = sigma_Y + part1(counter)-Deltabar2(counter);
    
    % Updating
    dDelta_s_t = (PostVar(sigma_Y,Vbar,tau)/sigma_Y^2).*(-Delta_s_t*dt+ones(size(Delta_s_t))*dZt(i));
    if i<Npre
        DELbias = (sum(biasvec(i:end))+sum(dZt(1:i)))/That;
    else
        DELbias = sum(dZt(i-Npre+1:i))/That;
    end
    Delta_s_t = [Delta_s_t(2:end)+dDelta_s_t(2:end)  DELbias];
    
    IntVec = [reduction*Part(2:end) bet*(1-reduction)*Xt2(counter)];
    
    counter = counter + 1;
end
Port=(part1-Deltabar2)/sigma_S + (sigma_Y/sigma_S)*(1-bet*fliplr(fhat)'./fst);

end

