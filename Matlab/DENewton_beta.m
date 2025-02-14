function [Y, infos] = DENewton_beta(D, w, pars)
%% 
% function [Y, infos] = DENewton_beta(D, w, []) % use default values
% function [Y, infos] = DENewton_beta(D, w, pars)
%
% Input: D -- Squared predistance matrix (diag(D) =0)
%        w -- w>0, a weight vector
% pars
%     pars.eig: 1 use eig (default)
%               0 use mexeig
%     pars.printyes:  0 no print out information (default)
%                     1 print out information
%     pars.scalew: = maxw/min(100, maxw),
%                    maxw = max(w) scale w between 0 and 100 (default)
%                  = 1 (no scaling)
%     pars.scaleD: = maxD/min(maxD, 100), 
%                    maxD = max(max(D)) force D in (0, 100) (default)
%                  1 (n scaling on D)
%     pars.y:  starting point, a vector of length(D)
%             default: y = zeors(n,1);
%     pars.tol: defaut 1.0e-6 
%
% Output:
%      Y -- Nearest Euclidean distance matrix from D
%
%    infos -- information about Y
%         infos.X -- Embedding points (in columns)
%               X is computed when pars.computingXyes = 1
%         infos.Z -- Z, Lagrange multiplier for Y \in \K^n_+ in the 
%                          original problem
%         infos.y -- Lagrange multiplier for \A(Y) = b in the 
%                          original problem (because \A = \diag().
%                KKT condition satisfies 
%                W^{1/2}(Y-D)W^{1/2} + diag(y) - Z = 0
%         infos.P -- Eigenvectors used in embedding
%         infos.lambda -- Eigenvalues used in embedding
%                   [P, lambda] = MYeig(-J_sYJ_s^T/2)
%                   s = w/sum(w) satisfying e^Ts = 1
%         infos.Pold -- from (-J^w(TD + Diag(y))J^w)
%         infos.lambdaold --             
%         infos.y  = the optimal Lagrange multiplier corresponding 
%                    to the diagonal constraints
%         infos.lambda = lambda;
%         infos.rank = Embedding_Dimension;
%         infos.Iter = k;
%         infos.feval = f_eval; number of function evaluations;
%         infos.t = time_used; total cpu used
%         infos.res = norm_b; norm of the gradient at the final iterate
%                             This is the residual one finally got.
%         infos.f = val_obj; final objective function value 
%%
%
%  This code is designed to solve %%%%%%%%%%%%%
%   min 0.5*\| Diag(w)^1/2 (X-D) Diag(w)^1/2 \|
%   s.t. X_ii =0, i=1,2,...,n
%        X is positive semidefinite on the subspace
%          \{x\in \Re^n: eTx =0 \}
% The W-weight is a vector
% This kind of problems arise from the proximal point method for for
% the H-weighted problem.
%%
%  Based on the algorithm  in %%%%%
%  ``A Semismooth Newton Method for 
%  the Nearest Euclidean Distance Matrix Problem'' 
%                By 
%             Houduo Qi                        
%   
%  First version date:   August 16, 2011
%  Second version date:  July    4, 2012  
%  This version data:    March  24, 2013
%%          
% Send your comments and suggestions to    %%%%%%
%        hdqi@soton.ac.uk      %%%%%%
%
% Acknowledgment: The code makes use of CorNewton.m developed by
% Houduo Qi and Defeng Sun (NUS) for computing 
% the nearest correlation matrix 
%
%%%%% Warning: Accuracy may not be guaranteed!!!!! %%%%%%%%
%%
%
t0 = tic;
n = length(D);
maxw = max(w);

%% Step 1: Check if it is the equal diagonal weight case
%
if ~isfield(pars, 'printyes')
    pars.printyes = 0; % information printed out
end
if ~isfield(pars, 'computingXyes')
    pars.computingXyes = 0; % no computing the embedding points X
end
if ~isfield(pars, 'y')
    pars.y = zeros(n,1); % default starting point
end
if ~isfield(pars, 'tol')
    pars.tol = 1.0e-6; % default tolerance
end
if ~isfield(pars, 'eig')
    pars.eig = 1; % use eig otherwise 0 use mexeig
end
if ~isfield(pars, 'positive_eig_level')
    pars.positive_eig_level = 1.0e-8; % positive eigenvalues less than 1.0e-8
                                      % are treated as zeros
end
positive_eig_level = pars.positive_eig_level;

equalweightflag = any(w/maxw - ones(n,1));
if equalweightflag == 0
    %fprintf('\n Equal diagonal weight: call ENewton_beta.m');
    [Y, infos] = ENewton_beta(D, pars); % 
    infos.Z   = maxw^2*infos.Z;
    infos.y   = maxw^2*infos.y;
    infos.res = maxw*infos.res;
    infos.f   = maxw^2*infos.f;
    return
end

%% Step 2: (otherwise) unequal diagonal weights, continue
%
if ~isfield(pars, 'scalew')
   pars.scalew = maxw/min(100, maxw); % wi \in (0, 100]
   %pars.scalew = maxw;
end
if ~isfield(pars, 'scaleD')
    maxD = max(max(D));
    pars.scaleD = maxD/min(100, maxD);
end

eigsolver = pars.eig;
error_tol = pars.tol;
prnt = pars.printyes;
scaleD = pars.scaleD;
scalew = pars.scalew;
y    = pars.y;
%%
% reset the data and to keep the old data
%
Dold = D;
wold = w;
scalefactor = scaleD*scalew;

D = D/scaleD;
w = w/scalew;
y = y./w;
y = y/scalefactor; % to scale the Lagrangian multiplier

sumw  = sum(w);
sqrtw = sqrt(w);

if prnt
  fprintf('\n ******************************************************** \n')
  fprintf( '        The Semismooth Newton-CG Method (DENewton.m)         ')
  fprintf('\n ******************************************************** \n')
  fprintf('\n The information of this problem is as follows: \n')
  fprintf(' Dim. of    sdp      constr  = %d \n',n)
end
%% Scale D so that D_{ij} \in [0, 1]
% Sfactor = scaleD
%
D       = -(D+D')/2; % make D symmetric
               % use -D instead because it is (-D) which is psd on e^\perp
TD      = (sqrtw*sqrtw').*D; % TD = \tilde(D)
%
error_tol = error_tol/(scaleD*scalew); % reset the tolerance level
if error_tol < 1.0e-9
   error_tol = 1.0e-7;
end

%
% calculate J(TD)J: J = I - 1/sum(w)*sqrt(w)*sqrt(w)'
% J = D - [ (D\sqt(w) \sqrt(w)' + sqrt(w) (D\sqrt(w))']/sum(w)
%     + (sqrt(w)'D\sqrt(w)/(sum(w)^2) \sqrt(w) \sqrt(w)'
%
TDw  = TD*sqrtw;
sumD = sqrtw'*TDw;
JDJ  = TDw*sqrtw';
JDJ  = (JDJ + JDJ')/sumw;
JDJ  = TD - JDJ + (sumD/sumw^2)*(sqrtw*sqrtw');
%%
k          = 0;
f_eval     = 0;
EigenD     = 0;
Iter_Whole = 200;
Iter_inner = 20; % Maximum number of Line Search in Newton method
maxit      = 200; %Maximum number of iterations in PCG

tol        = 1.0e-2; %relative accuracy for CGs
%
sigma_1=1.0e-4; %tolerance in the line search of the Newton method

prec_time = 0;
pcg_time  = 0;
eig_time  =0;

c = ones(n,1);
%M = diag(c); % Preconditioner to be updated
%
val_G = sum(sum(TD.*TD))/2;
%
% calculate Y = - J(D+diag(y))J
%
Y = JyJ(y, w);
Y = - (JDJ + Y);
Y = (Y+Y')/2;
%% eigen decomposition
 eig_time0  = tic;
 [P,lambda] = MYeig(Y,eigsolver); 
 eig_time = eig_time + toc(eig_time0); 
 EigenD   = EigenD + 1;
%%
 [f0,Fy] = gradient(y, TD, w, lambda, P);
 f       = f0;
 f_eval  = f_eval + 1; % number of function evaluations increased by 1
 b       = - Fy;
 norm_b  = norm(b);

Initial_f = val_G - f0;
 %
 if prnt
   fprintf('Initial Dual Objective Function value==== %d \n', Initial_f)
   fprintf('Newton: Norm of Gradient %d \n',norm_b)
 end
 
 Omega12 = omega_mat(lambda,n);
 x0 = y;

 tt = toc(t0);
[hh,mm,ss] = time(tt);

if prnt
  fprintf('\n   Iter.   Num. of CGs     Step length      Norm of gradient     time_used ')
  fprintf('\n    %d         %2.0d            %3.2e                      %3.2e         %d:%d:%d ',0,str2num('-'),str2num('-'),norm_b,hh,mm,ss)
end
%%

 while (norm_b>error_tol && k< Iter_Whole)

  prec_time0 = tic;
   c = precond_matrix(w,Omega12,P,n); % comment this line for  no preconditioning
  prec_time = prec_time + toc(prec_time0);
  
 pcg_time0 = tic;
 [d,flag,~,iterk]  = pre_cg(w,b,tol,maxit,c,Omega12,P,n);
 pcg_time = pcg_time + toc(pcg_time0);
 %d =b0-Fy; gradient direction
 %fprintf('Newton: Number of CG Iterations %d \n', iterk)
  
  if (flag~=0); % if CG is unsuccessful, use the negative gradient direction
     % d =b0-Fy;
     disp('..... Not a full Newton step......')
  end
 slope = (Fy)'*d; %%% nabla f d
%% 
    y = x0 + d; %temporary x0+d  
    Y = JyJ(y,w);
    Y = - (JDJ + Y);
    Y = (Y+Y')/2;
    %% eigen decomposition
 eig_time0  = tic;
 [P,lambda] = MYeig(Y,eigsolver); 
 eig_time = eig_time + toc(eig_time0); 
 EigenD   = EigenD + 1;
      
     [f,Fy] = gradient(y, TD, w, lambda, P); % increase of f_eval will be added
                                        % after the linear search  
     k_inner = 0;
     while(k_inner <=Iter_inner && f> f0 + sigma_1*0.5^k_inner*slope + 1.0e-6)
                           % line search procedure
        k_inner = k_inner+1;
        y       = x0 + 0.5^k_inner*d; % backtracking   
         
        Y = JyJ(y, w);
        Y = - (JDJ + Y);
        Y = (Y+Y')/2;
         
         %% eigen decomposition
 eig_time0  = tic;
 [P,lambda] = MYeig(Y,eigsolver); 
 eig_time = eig_time + toc(eig_time0); 
 EigenD   = EigenD + 1;
 
         [f,Fy] = gradient(y, TD, w, lambda, P);
      end % end of the line search procedure
      %
      if prnt
        if k_inner >=1
           fprintf('\n number of linear serach: %d', k_inner)
        end
      end
      f_eval = f_eval + k_inner + 1; % number of function evaluations
                                     % is increased over the line search
                                     % and the one before it
      x0 = y;
      f0 = f;
      
     k=k+1;
     b = -Fy;
     norm_b = norm(b);
     tt = toc(t0);
    [hh,mm,ss] = time(tt); 
  %   fprintf('Newton: Norm of Gradient %d \n',norm_b)
     if prnt   
        fprintf('\n   %2.0d         %2.0d             %3.2e          %3.2e         %d:%d:%d', ...
        k,iterk,0.5^k_inner,norm_b,hh,mm,ss)
     end
     
     Res_b(k) = norm_b;
    
     Omega12 = omega_mat(lambda,n);

 end %end loop for while i=1;
Ip = find(lambda > 0); % could set to 1.0e-7
In = find(lambda < -positive_eig_level); % could set to -1.0e-7
Embedding_Dimension = length(In);
  % The eigen-decomposition is on -J(D+\A^*y)J
  % The imbedding dimension is the number of negative eigenvalues of
  % -J(D+\A^*y)J.
  % This result is based on Haydan-Wells projection formula
  % see Eq. (37) in my paper.
r = length(Ip);
%
% use Y0 to store the matrix \Pi_{\S^n_+}(-Y)
%

if (r==0)
    Y0 = -Y;
    Y = TD + diag(y);
elseif (r==n)
    Y0 = zeros(n,n);
    Y = TD + diag(y) + Y;
elseif (r<=n/2)
    lambda1 = lambda(Ip);
    lambda1 = lambda1.^0.5;
    P1 = P(:, 1:r);
    P1 = P1*sparse(diag(lambda1));
    P1 = P1*P1';
       
    Y0 = P1 + Y;
    Y = TD+diag(y) + P1;% Optimal solution X* 
else 
    
    lambda2 = -lambda(r+1:n);
    lambda2 = lambda2.^0.5;
    P2 = P(:, r+1:n);
    P2 = P2*sparse(diag(lambda2));
    Y0 = P2*P2';
    Y = Y + Y0'; 
    Y = TD+diag(y) + Y;% Optimal solution X* 
end
 Y = (Y+Y')/2;
 % set diagonals of Y to zero
 %
 Y(1:(n+1):end) = 0;
%
% computing the objective (both primal and dual) for the reformulated
% problem
dual_f = val_G - f;
primal_f = 0.5*sum(sum((( Y - TD).^2)));
%Z = Y - TD - diag(y);
%norm(Z)
%
%  val_obj = sum(sum((Y-TD).*(Y-TD)))/2;
%%
% Information on Y, y, Z for the reformulated problem on K^n_w
% this w = wold/scalew;
% Y and y have been obtained above
% calculate Z
%
Z = -TD + Y - diag(y); % -TD = W^{1/2}*(Dold/scaleD)*W^{1/2};
Z = -Z; % = -Y + TD + diag(y)
%comp_gap = sum(sum(Y.*Z)); % complementarity gap trace(YZ) = 0 in theory
% 
% output eigeninformation from Y
%
infos.Pold = P;           %[P, lambda] = My(-J^w(TD+ Diag(y))J^w))
infos.lambdaold = lambda; %this is in case for future use

%% Actual embedding points (obtained from Y0): 1st way
%
computingXyes = pars.computingXyes;

%if computingXyes
%   w1 = 1./sqrtw;
%   Y0 = (w1*w1').*Y0;
%   Y0 = (Y0+Y0')/2;
%   Y0 = full(Y0);
%   [P0, lambda0] = MYeig(Y0, eigsolver);
%   lambda0 = scaleD*lambda0/2;
% 
% % output information P0, lambda0
%   infos.P = P0;           %[P0, lambda0] = MYeig(-J_sYJ_s^T) where Y is to
%   infos.lambda = lambda0; % be calculated below.
% %
%   r = sum(lambda0>0);
%   lambda0 = lambda0(1:r).^(1/2);
%   X = P0(:, 1:r)*sparse(diag(lambda0));
%   X = X';
%   infos.X = X;
%end
%%%%%%% End of 1st way of computing X

Y = abs(Y); % put (-1) sign back to Y and abs ensure Yij >= 0

% return Y, y, Z for the original problem on K^n_+ with Dold, wold
%
sqrtw = sqrt(wold);
w1 = 1./sqrtw;

Y = scalefactor*((w1*w1').*Y);
y = scalefactor*(wold.*y);
Z = scalefactor*((sqrtw*sqrtw').*Z);

infos.y = y;
infos.Z = Z;
%%
% the output Y, y, Z satisfies in theory
% (wold*wold').*(Y - Dold) + diag(y) - Z = 0 %dual feasibility
%  check the dual feasibility
%
Z = (wold*wold').*(Y - Dold) +  diag(y) - Z;
dfeasi = sum(sum(Z.*Z));
dfeasi = sqrt(dfeasi); % should be very small
infos.dfeasi = dfeasi;
%%
norm_b      = scalefactor*norm_b;

% calculate the final objective function value
Dold = (sqrtw*sqrtw').*(Y-Dold);
val_obj = 0.5*sum(sum((Dold.^2)));

%%%%%%%%%%%%%%% 2nd way: direct decompsoition of -0.5JYJ %%%%%%%%%%%%%
% Comment: the resulting distance matrix from X is more
%          accurate than that obtained by the 1st way
% verify: diff = squareform(pdist(X')) - Y
%         normdiff = norm(diff, 'fro');
%         this is in the order of 1.0e-5
if computingXyes
   z = sum(Y, 2);
   sumY = sum(z);
   Z = z*ones(1,n);
   Z = Y - (Z+Z')/n + sumY/n^2;
   Z = -0.25*(Z+Z'); % 0.25 is used because 0.5 for symmetry 
                     % and 0.5 for -0.5JYJ

   eig_time0 = tic;
  [P,lambda] = MYeig(Z,eigsolver); 
  eig_time = eig_time + toc(eig_time0); 
  EigenD   = EigenD + 1;
  infos.P = P;
  infos.lambda = lambda;

  r = sum(lambda > positive_eig_level);
  Embedding_Dimension = r;
  P = P(:, 1:r);
  lambda = lambda(1:r).^(1/2);
  P = P*sparse(diag(lambda));

  infos.X = P';
  infos.EigD = EigenD;
end
%%%%%%%%%%%%%%%%%% End of 2nd way %%%%%%%%%%%%%%%%%%%%%%
 
 time_used = toc(t0);
% fprintf('\n')
% More output information
infos.rank   = Embedding_Dimension;
infos.Iter   = k;
infos.feval  = f_eval;
infos.t      = time_used;
infos.res    = norm_b;
infos.f      = val_obj;
infos.EigenD = EigenD;
%
if prnt
fprintf('\n\n')
fprintf('Norm of Gradient (in unscaled space) %d \n', full(norm_b))
fprintf('Number of Iterations == %d \n', k)
fprintf('Number of Function Evaluations == %d \n', f_eval)
fprintf('Objective value for the Dual Problem ========== %d \n', full(dual_f))
fprintf('Objective value for the Primal Problem ======== %d \n', full(primal_f))
% for the solved dual problem
fprintf('Final Original Objective Function value ======= %d \n', full(val_obj))
% for the reformulated problem: gap = dual_f - primal_f should be small
fprintf('Embedding dimension ================= %d \n',Embedding_Dimension)
fprintf('Computing time for computing preconditioners == %d \n', prec_time)
fprintf('Computing time for linear systems solving (cgs time) ====%d \n', pcg_time)
 if eigsolver
    fprintf('Computing time for  eigenvalue decompostions (calling eig time)==%d \n', eig_time)
 else
     fprintf('Computing time for  eigenvalue decompostions (calling mexeig time)==%d \n', eig_time)
 end
fprintf('Total computing time (in s) ==== =====================%d \n',time_used)
end
% 


%%% end of the main program

%%% To change the format of time 
function [h,m,s] = time(t)
t = round(t); 
h = floor(t/3600);
m = floor(rem(t,3600)/60);
s = rem(rem(t,60),60);
return
%%% End of time.m

%%%%%% To generate J*diag(y)J
%
function Y = JyJ(y, w)

sumw = sum(w);
t    = w'*y;
w    = sqrt(w); %w is positive; save using a new variable for sqrt(w)

Y = - (y.*w)*w';
Y = (Y+Y')/sumw;
Y = Y + diag(y) + t/sumw^2*(w*w');

return
%%%%%%%%%%%
%%% mexeig decomposition
function [P,lambda] = MYeig(X,eigsolver)
X = full(X);
if eigsolver %% use matlab built-in function eig
     [P, Lambda] =  eig(X);   %% X= P*diag(D)*P'
     P = real(P);
     lambda = real(diag(Lambda));
    
else %% use mexeig developed by Prof. Defeng Sun
    [P,lambda] = mexeig(X);
    P          = real(P);
    lambda     = real(lambda);    
    
end
% make sure eigenvalues are in decreasing order
if issorted(lambda)
    lambda = lambda(end:-1:1);
    P      = P(:,end:-1:1);
elseif issorted(lambda(end:-1:1))
    return;
else
    [lambda, Inx] = sort(lambda,'descend');
    P = P(:,Inx);
end
return
%%% End of MYeig.m
%%%%%%
%%%%%% To generate F(y) %%%%%%%
%%%%%%%

function [f,Fy]= gradient(y,D,w,lambda,P)
 
n  = length(lambda);
f  = 0.0;
Fy = zeros(n,1);
 
%lambdap=max(0,lambda);
%H =diag(lambdap); %% H =P^T* H^0.5*H^0.5 *P
%% Compute Fy
r = sum(lambda>0);
lambda1 = lambda(1:r).^0.5;
P= P(:,1:r)';
for i = 1:r;
    P(i,:) = lambda1(i)*P(i,:);
end
 
 i=1;
 while (i<=n)
       Fy(i) = P(:,i)'*P(:,i) + y(i);
       i=i+1;     
 end
  
 Dy = D + diag(y);
%
%
r = sum(lambda< 0); % negative eigenvalues
lambdan = lambda(end:-1:(n-r+1));
f = f + lambdan'*lambdan;

%
% the rest part depends on Q
%
sumw = sum(w);
Dyw = Dy*sqrt(w); % Dy*sqrt(w)
%Dyw_old = Dyw;
%
% to save using a new variable yhat (see below)
%
% c = ( sqrt(w)'*Dyw + sqrt(sumw)*Dyw(end) )/( sumw + sqrt(sumw)*sqrt(w(end)));
% Dyw = (-Dyw + c*sqrt(w))/sqrt(sumw); % denoted by Dyw (Dyw not to be used any more)
% Dyw(end) = Dyw(end) + c;
%  
%  f0 = f + sum(Dyw.^2) + sum(Dyw(1:end-1).^2);
%  f0/2
% use a new way to calculate it
sqaa0 = (Dyw'*Dyw)/sumw;
a0    = (sqrt(w)'*Dyw)/sumw;
f     = f + 2*sqaa0 - a0^2;
%
f = 0.5*f;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% end of gradient.m %%%%%%

%%%%%%%%%%%%%%        %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% To generate the first -order difference of lambda
%%%%%%%

%%%%%%%%%%%%%%        %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% To generate the essential part of the first -order difference of d
%%%%%%%
function Omega12 = omega_mat(lambda,n)
%We compute omega only for 1<=|idx|<=n-1
idx.idp = find(lambda>0);
idx.idm = setdiff([1:n],idx.idp);
n =length(lambda);
r = length(idx.idp);
 
if ~isempty(idx.idp)
    if (r == n)
        Omega12 = ones(n,n);
    else
        s = n-r;
        dp = lambda(1:r);
        dn = lambda(r+1:n);
        Omega12 = (dp*ones(1,s))./(abs(dp)*ones(1,s) + ones(r,1)*abs(dn'));
        %  Omega12 = max(1e-15,Omega12);

    end
else
    Omega12 =[];
end

    %%***** perturbation *****
    return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% end of omega_mat.m %%%%%%%%%%

%%%%%% PCG method %%%%%%%
%%%%%%% This is exactly the algorithm by  Hestenes and Stiefel (1952)
%%%%%An iterative method to solve A(x) =b  
%%%%%The symmetric positive definite matrix M is a
%%%%%%%%% preconditioner for A. 
%%%%%%  See Pages 527 and 534 of Golub and va Loan (1996)

function [p,flag,relres,iterk] = pre_cg(w,b,tol,maxit,c,Omega12,P,n)
% Initializations
r = b;  %We take the initial guess x0=0 to save time in calculating A(x0) 
n2b =norm(b);    % norm of b
tolb = tol * n2b;  % relative tolerance 
p = zeros(n,1);
flag=1;
iterk =0;
relres=1000; %%% To give a big value on relres
% Precondition 
z =r./c;  %%%%% z = M\r; here M =diag(c); if M is not the identity matrix 
rz1 = r'*z; 
rz2 = 1; 
d = z;
% CG iteration
for k = 1:maxit
   if k > 1
       beta = rz1/rz2;
       d = z + beta*d;
   end
   %ww= Jacobian_matrix(d,Omega,P,n); %w = A(d); 
   ww = Jacobian_matrix(d,w,Omega12,P); % W =A(d)
   denom = d'*ww;
   iterk =k;
   relres = norm(r)/n2b;              %relative residue = norm(r) / norm(b)
   if denom <= 0 
       sssss=0
       p = d/norm(d); % d is not a descent direction
       break % exit
   else
       alpha = rz1/denom;
       p = p + alpha*d;
       r = r - alpha*ww;
   end
   z = r./c; %  z = M\r; here M =diag(c); if M is not the identity matrix ;
   if norm(r) <= tolb % Exit if Hp=b solved within the relative tolerance
       iterk =k;
       relres = norm(r)/n2b;          %relative residue =norm(r) / norm(b)
       flag =0;
       break
   end
   rz2 = rz1;
   rz1 = r'*z;
end

return

%%%%%%%% %%%%%%%%%%%%%%%
%%% end of pre_cg.m%%%%%%%%%%%


%%% To generate the Jacobian product with x: F'(y)(x)
function Ax = Jacobian_matrix(x, w, Omega12, P)

n = length(w);
[r,s]  = size(Omega12); 
Ax = zeros(n,1);

if (r==0)
    Ax = (1+1.0e-10)*x;
    return
elseif (r==n)
    sumw = sum(w);
    Ax = (2/sumw)*(x.*w) - (x'*w/sumw^2)*w;
    Ax = Ax + 1.0e-10*x;
    return
end
%
% constants to be used
%
pw = (P'*sqrt(w))/sum(w);
xw = x.*sqrt(w);
px = P'*xw;
sumxw = x'*w;
%

P1 = P(:,1:r);
P2 = P(:,r+1:n);
    
px = 0.5*sumxw*pw - px; % replace startx by px
PX = P'.*px(:, ones(n,1));
PY = P'.*pw(:, ones(n,1));
    
if (r<n/2)
   H1 = P1'*sparse(diag(x));
   Omega12_old = Omega12;
   Omega12 = Omega12.*(H1*P2);
   H = [(H1*P1)*P1' + Omega12*P2'; Omega12'*P1'];

   PX12 = Omega12_old'*PX(1:r, :);
   PY12 = PY(1:r, :)'*Omega12_old;
%%%%% 1st way to calculate Ax           
%             PY12 = PY12'.*PX(r+1:n, :);
%             PX12 = PY(r+1:n, :).*PX12;
%             
%              Ax = sum(PY(1:r, :)).*sum(PX(1:r, :)) + sum(PX12+PY12);
%              Ax = 2*Ax + sum(P'.*H);
%              Ax = Ax';
%              Ax = (1+1.0e-10)*x - Ax;
%%%%%%
%%%%%% 2nd way to calculate Ax
    i=1;
    while (i<=n)
         Ax(i) = P(i,:)*H(:,i); % part from the digonal part
         v  = sum(PY(1:r, i))*sum(PX(1:r, i)) ...
            + PY12(i, :)*PX(r+1:n, i) + PY(r+1:n, i)'*PX12(:, i);
         Ax(i) = x(i) - Ax(i) - 2*v;  
         i=i+1;
    end 
%%%%%%%%%%%%% end of 2nd way
else % if r>=n/2, use a complementary formula.
    %H = ((E-Omega).*(P'*Z*P))*P';               
    H2 = P2'*sparse(diag(x));
    Omega12 = ones(r,s)- Omega12;
    Omega12_old = Omega12;
    Omega12 = Omega12.*((H2*P1)');
    H = [Omega12*P2'; Omega12'*P1' + (H2*P2)*P2'];
           
    PX12 = Omega12_old*PX(r+1:n, :);
    PY12 = PY(r+1:n, :)'*(Omega12_old');
%%%%% 1st way to calculate Ax            
%             PX12 = PY(1:r, :).*PX12;
%             PY12 = PY12'.*PX(1:r, :);
%             
%             Ax = sum(PY(r+1:n, :)).*sum(PX(r+1:n, :)) + sum(PX12+PY12);
%             Ax = 2*Ax + sum(P'.*H);
%             Ax = Ax';
%%%%% 2nd way to calculate Ax 
    i=1;
    while (i<=n)
        % Ax(i) = x(i) - P(i,:)*H(:,i); %from diagonal part
         Ax(i) = P(i,:)*H(:,i);
            v  = sum(PY(r+1:n, i))*sum(PX(r+1:n, i)) ...
               + PY(1:r, i)'*PX12(:, i) + PY12(i, :)*PX(1:r, i);
         Ax(i) = Ax(i) + 2*v;  
         i=i+1;
    end
%%%%%% end of 2nd way
    sumw = sum(w);
    Ax = (2/sumw)*(x.*w) - (x'*w/sumw^2)*w + Ax;
end

  Ax = Ax + 1.0e-10*x;
    return
%%% End of Jacobian_matrix.m  
   
   

%%%%%% To generate the diagonal preconditioner%%%%%%%
%%%%%%%

function c = precond_matrix(w,Omega12,P,n)

[r,s] = size(Omega12);
c     = ones(n,1);
sumw  = sum(w);

if (r==0) % Omega = 0
    c = ones(n,1);
    return
elseif (r==n) % Omega = E
    t = w./sumw;
    c = 2*t - t.^2;
    return
end

 H = P';
 h = H*sqrt(w)/sumw; %average row sum of P'
 H = H.*(H - h*sqrt(w)');

if (r >0 && r < n/2) % (0 < r < n/2)
     H12 = H(1:r,:)'*Omega12;
     for i=1:n
        c(i) = (sum(H(1:r,i)))^2;
        c(i) = c(i) + 2.0*(H12(i,:)*H(r+1:n,i));
        c(i) = 1 - c(i);
     end
else  % if r<n% if r>=n/2, use a complementary formula     
        Omega12 = ones(r,s)-Omega12;
        H12 = Omega12*H(r+1:n,:);        
        for i=1:n
            c(i) = (sum(H(r+1:n,i)))^2;
            c(i) = c(i) + 2.0*(H(1:r,i)'*H12(:,i));
        end
        t = w./sumw;
        c = c + 2*t - t.^2; % including the case r == n.
end

 c = max(1.0e-8, c);

return

 
%%%%%%%%%%%%%%%
%end of precond_matrix.m%%%

%



