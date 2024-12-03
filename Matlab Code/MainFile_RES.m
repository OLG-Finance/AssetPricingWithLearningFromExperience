% Main code for the numerical resuls in "Asset Prices and Portfolio Choice with Learning from Experience" %%
% By Paul Ehling, Alessandro Graniero, and Christian Heyerdahl-Larsen 
% This version: August 2017.

clear all;

%Parameters
rho     = 0.001;   % Time discount rate 
nu      = 0.02;    % Death rate 
mu_Y    = 0.02;    % Growth rate of output
sigma_Y = 0.033;   % Standard deviation of output
sigma_S = sigma_Y; % In equilibrium the stock price diffusion is the same as output diffusion
w       = 0.92;    % Fraction of total output paid out as dividend



% Some pre-calculations
D = rho^2+4*(rho*nu+nu^2)*(1-w);
bet = (rho+2*nu-sqrt(D))/(2*nu);
rlog = rho + mu_Y-sigma_Y^2;

% Setting prior variance  
That = 20;                   % Pre-trading period
dt = 1/12;
Npre = That/dt;
Vbar    = (sigma_Y^2)/That;  % Prior variance 
Tcohort = 500;               % Time horizon to keep track of cohorts
Nt = Tcohort/dt; 


MC = 1; %could draw multiple build ups. 
fMAT = zeros(MC,Nt);
for i=1:MC
    dZt = sqrt(dt)*randn(Nt-1,1);
    [Deltabar,IntVec,Xt,Delta_s_t,Yt,Zt,f,tau] = BuildUpCohortsMAIN(dZt,Nt,dt,rho,nu,Vbar,mu_Y,sigma_Y,bet,That);
    fMAT(i,:) = f;
end

% Initializing some variables
Mpaths =100;
Tsample=Tcohort/100;
Nsamples=100;
stepCorr =Tsample*12;
corrZport = zeros(Mpaths,Nsamples);
corrZMUs_t = zeros(Mpaths,Nsamples);
corrMU_sMUs_t = zeros(Mpaths,Nsamples);
corrMuSmuHat = zeros(Mpaths,1);
fMAT  = zeros(Mpaths,Nt);
mC = zeros(Mpaths,Nt);
sC = zeros(Mpaths,Nt);
DeltaHatMAT = zeros(Mpaths,Nt);
rMAT = zeros(Mpaths,Nt);
thetaMAT = zeros(Mpaths,Nt);
portMAT = zeros(Mpaths,Nt);
Zmat = zeros(Mpaths,Nt);

% Expected returns
muSMAT = zeros(Mpaths,Nt);      % Expected returns under true measure
muSsMAT = zeros(Mpaths,Nt);     % Expected returns under the measure of the agent we track
muShatMAT = zeros(Mpaths,Nt);   % Simple average of expected returns, or consensus belief
EtMAT = zeros(Mpaths,Nt);
VtMAT = zeros(Mpaths,Nt);
RxMAT = zeros(Mpaths,Nt);
muCst = zeros(Mpaths,Nsamples);
logmuCst = zeros(Mpaths,Nsamples);
sigCst = zeros(Mpaths,Nsamples);
stdCst = zeros(Mpaths,Nsamples);

%% This is the main loop. First, it builds up the economy with a large number of cohorts. Second, it simulates the stationary economy forward.

counter2 = 1;
for k=1:Mpaths
    k 
    
    % This loop characterizes the build-up period with large number of cohorts (Nt).
    if counter2==10   % Every 10th path we re-draw the build up period (going from 1 to 6000 cohorts)
        dZt = sqrt(dt)*randn(Nt-1,1);
        [Deltabar,IntVec,Xt,Delta_s_t,Yt,Zt,f,tau] = BuildUpCohortsMAIN(dZt,Nt,dt,rho,nu,Vbar,mu_Y,sigma_Y,bet,That);
        counter2 = 1;
    end
    dZforbias = diff(Zt);
    biasvec = dZforbias(end-Npre+1:end);
    dZt = sqrt(dt)*randn(Nt,1);
    Zt = cumsum(dZt);   
    
    % This function simulates the economy forward. We initialize the variables to be simulated using 
    % the values computed in the build-up period. 
   
    [Xt2,Deltabar2,Part1,mu_S,mu_S_t,muhat_S_t,r_t,theta_t,Port,muC_s_t,sigmaC_s_t,BIGf,BIGDELTA,Et,Vt,dR] = SimCohortsMAIN(biasvec,dZt,Nt,tau,IntVec,Delta_s_t,dt,rho,nu,Vbar,mu_Y,sigma_Y,sigma_S,bet,That,Npre);
    
    RxMAT(k,:) = dR';
    EtMAT(k,:) = Et';
    VtMAT(k,:) = Vt';
    DeltaHatMAT(k,:) = Deltabar2';
    rMAT(k,:) = r_t';
    thetaMAT(k,:) = theta_t';
    Zmat(k,:) = Zt(1:Nt)';
  
    portMAT(k,:) = Port';
    
    muSMAT(k,:) = mu_S'+rlog-r_t';
    muSsMAT(k,:) = mu_S_t'+rlog-r_t';
    muShatMAT(k,:) = muhat_S_t'+rlog-r_t';
   
    mu_S = mu_S+rlog-r_t;
    muhat_S_t = muhat_S_t+rlog-r_t;
    mu_S_t =  mu_S_t+rlog-r_t;
    mC(k,:) = muC_s_t';
    sC(k,:) = sigmaC_s_t';
    corrMuSmuHat(k) = corr(muhat_S_t,mu_S);
    fMAT(k,:) = mean(BIGf);
    
    % The following loop stacks a large number (Nsamples) of simulations of the economy into matrices.
    for l=1:Nsamples
        corrZMUs_t(k,l)     = corr(Zt((1+(l-1)*stepCorr):l*stepCorr),mu_S_t((1+(l-1)*stepCorr):l*stepCorr));   
        corrZport(k,l)     = corr(Zt((1+(l-1)*stepCorr):l*stepCorr),Port((1+(l-1)*stepCorr):l*stepCorr));
        corrMU_sMUs_t(k,l) = corr(mu_S((1+(l-1)*stepCorr):l*stepCorr),mu_S_t((1+(l-1)*stepCorr):l*stepCorr));
        muCst(k,l) = mean(muC_s_t(1+(l-1)*stepCorr:l*stepCorr));
        logmuCst(k,l) = mean(muC_s_t(1+(l-1)*stepCorr:l*stepCorr)'-0.5*(sigmaC_s_t(1+(l-1)*stepCorr:l*stepCorr)').^2);
        sigCst(k,l) = mean(sigmaC_s_t(1+(l-1)*stepCorr:l*stepCorr));
        stdCst(k,l) = mean(abs(sigmaC_s_t(1+(l-1)*stepCorr:l*stepCorr)));
    end
    counter2 = counter2+1;
end


MaxAge = 100;
MaxAgeN = MaxAge/5;
tperiod=Tsample:Tsample:100;
meanZport = mean(corrZport);
meanZmus_t = mean(corrZMUs_t);

% Compute mean values from the simluations
meanMus = mean(corrMU_sMUs_t);
meanMuCst = mean(muCst);
meanSCst = mean(sigCst);
meanStdCst = mean(stdCst);
meanLogMuCst = mean(logmuCst);

%% FIGURE 1, FIGURE 2 AND FIGURE 3 OF SECTION 3.

% FIGURE 1 in the paper
figure;
subplot(1,2,1);
plot(tperiod,meanMus(1:MaxAgeN));
xlabel('Age')
ylabel('corr( \mu^S_t - r_t, \mu^S_{s,t} - r_t )')
subplot(1,2,2);
plot(tperiod,meanZmus_t(1:MaxAgeN));
xlabel('Age')
ylabel('corr( z_t, \mu^S_t - r_t )')

% FIGURE 2 in the paper
figure;
plot(tperiod,meanZport(1:MaxAgeN));
xlabel('Age')
ylabel('corr( z_t, \pi_{s,t} )')

% FIGURE 3 in the paper
figure;
subplot(2,1,1);
plot(tperiod,meanLogMuCst(1:MaxAgeN));
xlabel('Age')
ylabel('Drift of log consumption')
subplot(2,1,2);
plot(tperiod,meanStdCst(1:MaxAgeN));
xlabel('Age')
ylabel('Volatility of log consumption growth')



% Mean values of the diffusion of the market view
disp('Mean value of market view diffusion');
mean(mean(VtMAT+EtMAT))
disp('Relative importance');
mean(mean(EtMAT))/mean(mean(VtMAT+EtMAT))
mean(mean(EtMAT./(VtMAT+EtMAT)))

% Correlations and Stdvs
stdRPtrue = zeros(Mpaths,1);
stdRPsurvey = zeros(Mpaths,1);
corrRPtruesurvey = zeros(Mpaths,1);
for k=1:Mpaths
    mutrue = muSMAT(k,:)';
    muSurvey = muShatMAT(k,:)';
    stdRPtrue(k) = std(mutrue);
    stdRPsurvey(k) = std(muSurvey);
    corrRPtruesurvey(k) = corr(mutrue,muSurvey);
end
disp('Std of true RP');
mean(stdRPtrue)
disp('Std of survey RP');
mean(stdRPsurvey)
disp('Ratio of std survey/std true');
mean(stdRPsurvey./stdRPtrue) % The difference between this and next is Jensen's term
mean(stdRPsurvey)/mean(stdRPtrue)
disp('Corr RP true and RP survey');
mean(corrRPtruesurvey)


%% Setting up the regressions performed in TABLE 1

% Regressions in TABLE 1
RegsExtrapSurvey = zeros(Mpaths,5);
RegsExtrapTrue   = zeros(Mpaths,5);
RegsRPtrueSurvey = zeros(Mpaths,5);

% Creating 12 month (overlapping) returns
RMAT12 = zeros(Mpaths,Nt-13);
for j=1:Nt-12
      RMAT12(:,j) = sum(RxMAT(:,j+1:j+12)+rMAT(:,j:j+11),2);
end

% Running regressions for each path
for i=1:Mpaths
    R12 = RMAT12(i,1:end-1)';
    mS = (muShatMAT(i,:)'+nu*(1-bet))*dt*12;  %survey mean
    mT = (muSMAT(i,:)'+nu*(1-bet))*dt*12; %true mean     
     
    X = [ones(size(R12)) R12];
    [bv,sebv,R2v,R2vadj,v,F] = olsgmm(mT(14:end),X,1,1);
    t = bv./sebv;
    RegsExtrapTrue(i,1:2) = bv';
    RegsExtrapTrue(i,3:4) = t';
    RegsExtrapTrue(i,5) = R2v;
     
    [bv,sebv,R2v,R2vadj,v,F] = olsgmm(mS(14:end),X,1,1);
    t = bv./sebv;
    RegsExtrapSurvey(i,1:2) = bv';
    RegsExtrapSurvey(i,3:4) = t';
    RegsExtrapSurvey(i,5) = R2v;   
    
    X = [ones(size(mS)) mS];
    [bv,sebv,R2v,R2vadj,v,F] = olsgmm(mT,X,1,1);
    t = bv./sebv;
    RegsRPtrueSurvey(i,1:2) = bv';
    RegsRPtrueSurvey(i,3:4) = t';
    RegsRPtrueSurvey(i,5) = R2v;
end


% Table 1 results:

%Reg (1)
disp('Reg (1) - beta');
mean(RegsExtrapSurvey(:,2))
disp('Reg (1) - t-stat');
mean(RegsExtrapSurvey(:,4))
disp('Reg (1) - R2');
mean(RegsExtrapSurvey(:,end))
%Reg (2)
disp('Reg (2) - beta');
mean(RegsExtrapTrue(:,2))
disp('Reg (2) - t-stat');
mean(RegsExtrapTrue(:,4))
disp('Reg (2) - R2');
mean(RegsExtrapTrue(:,end))
%Reg (3)
disp('Reg (3) - beta');
mean(RegsRPtrueSurvey(:,2))
disp('Reg (3) - t-stat');
mean(RegsRPtrueSurvey(:,4))
disp('Reg (3) - R2');
mean(RegsRPtrueSurvey(:,end))

