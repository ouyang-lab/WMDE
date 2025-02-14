
function [X, infos] = EMBED(D, dim, pars)
%%
% EMBED: EMbedding the Best Euclidean Distances
%
% This code aims to recover the unknow sensors based on the avaialble
% distance information. 
%
% Refinement step (Step 4 in the code) is taken from Kim-Chaun Toh SNLSDP
% solver
%
% Input:
%  
% D = [anchor-anchor (squared) distance, anchor-sensor (squared) distance;
%      sensor-anchor (squared) distance, sensor-sensor (squared) distance]
%      distances are SQUARED    
%      diag(D) = 0
%              Note: Format of D is different from SNLSDP of Kim-Chuan Toh
% dim:  The embedding dimensions (e.g. dim = 2 or 3)
%
% pars: parameters and other information
%
% pars.m  = m -- the number of known anchors, =0 if none.
% pars.PP = PP = [PA, PS] -- Positions of Ancors/Sensors
%           PA -- r-by-m matrix of coordinates of m anchors
%           PS -- r-by-(n-m) matrix of coordiates of (n-m) sensors
%           PS may be r-by-n1 matrix of coordinates of n1 sensors (n1<n-m)
%           (not sure if this case has any application)
% pars.DD = [sensor-sensor distance, sensor-anchor distance]
%         This format is to use Toh's refinepositions.m (Refinement Step)
%         distances are NOT squared
%         If not provided, DD will be calculated from D
% pars.H -- H weights (default: H =spones(D))
% pars.I
% pars.J
% pars.b: for fixed distances Dij = bij for (i, j) \in I x J (i not= j)
%         among sensor-anchor/sensor distances
% pars.anchorconstraintyes = 1 distances between anchors are hard enforced
%                              as equality constraints
%                            0 No hard constraints on anchors (default)
%                              Instead, Hij = 100 between ai and aj
%
% pars.EigenRatioLevel = the percent that has already been accounted for
%                        by the first (dim) eigenvalues
%                        sum(lambda(1:dim))/sum(lambda)
%                        90% (default)
%                        This is used to stop the algorithm from computing
%                        too many iterations that only force the small
%                        eigenvlaues smaller.
% pars.pcentmissing = p: tolerance level of pecentage of missing distances
%                        to use the shortest paths to replace them
%                     0 (default) if missing distances exist, use shortest
%                     10% (e.g)
% pars.spathyes: = 1 shortest paths to replace the missing distances
%                  0 do not use shortest paths.
%                  It is calculated by graphallshorestpaths
% pars.refinement: 1 to include the refinement step of Toh et al.
%                  0 otherwise
%                  If I = [] 
%                  (there are no fixed sensor-anchor/sensor distances)
%                  set it to 1; otherwise, set it to 0
% pars.plotyes: 1 plot localizations in Re^2 or 3
%                0 no plot
% pars.r -- Embedding dimension in Emap algorithms (r >= dim)
%           default: r = dim
%
% Output
%
% X -- final localizations of the sensors
% 
% infos:
% 
% infos.Y -- the best EDM Y with the required embedding dimension (dim)
% infos.f -- \| Ho(Y-D)\| (objective value)
% infos.RMSD -- Root Mean Square Distance if PP is not empty
%               [] if PP =[];
% infos.t -- Total computing time
%%
% Based on the paper:
% Computing the nearest Euclidean distance matrix with low embedding
% dimensions 
% by Hou-Duo Qi and Xiaoming Yuan (2012)
%
% Send your comments and suggestions to    %%%%%%
%          hdqi@soton.ac.uk                %%%%%%
%
%%%%% Warning: Accuracy may not be guaranteed!!!!! %%%%%%%%

%%%%%%%%% This version: April 14, 2013     %%%%%%%

%% Step 0: Read data
% 
t0 = tic;
n  = length(D);

if isfield(pars, 'm')
    m = pars.m;
else
    m = 0;
end
if isempty(m) % this is for the case that pars.m = [];
    m =0;
end

% np is the number of known anchors and known sensors
if isfield(pars, 'PP') % if m>0, it is not empty
    PP     = pars.PP;
    [~, np] = size(PP); % np may not equal n as not all positions are known
else
    PP = [];
    np =0;
end

if isfield(pars, 'I')
    I = pars.I;
    J = pars.J;
    b = pars.b;
    n0 = length(I);
else
    I =[];
    n0 = 0;
end

if ~isfield(pars, 'anchorconstraintyes')
    pars.anchorconstraintyes = 0;
end
anchorconstraintyes = pars.anchorconstraintyes;

if (anchorconstraintyes)
 if (m >= 2)
      m2 = m*(m-1)/2;
      II = [0, cumsum((m-1):-1:1)];
      I1 = zeros(m2, 1);
      J1 = I1;
      b1 = I1;
      for i = 1:(m-1)
          I1(II(i)+1:II(i+1)) = i*ones(m-i, 1);
          J1(II(i)+1:II(i+1)) = (i+1:m)';
          b1(II(i)+1:II(i+1)) = D(i, (i+1):m)'; %all column vectors
      end
    
      if (n0 == 0)
         I = I1; J = J1; b = b1;
      else
         I = [I1; I]; J = [J1; J]; b = [b1; b];
      end
    
      n0     = length(I);
      pars.I = I;
      pars.J = J;
      pars.b = b;
 end
end

% H weights
if ~isfield(pars, 'H')
    pars.H = spones(D);
end
H = pars.H;

if (anchorconstraintyes == 0) && (m >= 2) % more than 2 anchors
   H(1:m, 1:m) = 100*H(1:m, 1:m);
   pars.H = H;
end

h = sum(H,2);

if any(h==0)
    error('the graph is not connected!');
end

% to use the shortest paths for missing distances
if ~isfield(pars, 'spathyes')
    pars.spathyes = 1;
end
spathyes = pars.spathyes;

Dold = D;

D = sqrt(D);

% Define DD for refinepositions.m of Toh
if ~isfield(pars, 'DD')
    pars.DD = [D((m+1):end, (m+1):end), D( (m+1):end, 1:m )];
end
DD = pars.DD;

% percentage of missing distances to control to use shortestpaths
if ~isfield(pars, 'pcentmissing') 
    pars.pcentmissing = 0;
end
pcent = pars.pcentmissing;

if ~isfield(pars, 'EigenRatioLevel')
    pars.EigenRatioLevel = 90/100;
end

% not squared distances used for the shortest path
if (spathyes) && (nnz(D)/(n^2-n) < (1-pcent))
    fprintf('\n ^^^^^^^ EMBED: Computing the shortest paths ... ^^^^^^^');
    D = graphallshortestpaths(D);
    D = max(D.^2, Dold); % to only replace those missing distances
else
    D = Dold; % no shortest paths are used
end
pars.spathyes = 0;

% plot parameters
if ~isfield(pars, 'plotyes')
    pars.plotyes = 0; % no plot
end

plotyes = pars.plotyes;


% embedding dimensions in Emap algorithms
if ~isfield(pars, 'r')
    pars.r = dim;
end

r = pars.r;
%% Step 2: call EMap algorithms
%
pars.tol = 1.0e-2;  % loose tolerance to speed up the computation
pars.tol2 = 1.0e-1; % loose tolerance for the off-diagonal constraints
pars.printyes = 0;
[Y, infos] = rHENewton2_beta(D, pars);

X = infos.X;
X = X(1:dim, :);   % take the first (dim)-dimensions in X, which may have
                   % more dimensions than (dim).
% if r > dim % the embedding dimension in EMap is bigger than dim
%    X = infos.X;
%    X = X(1:dim, :);
% else
%    X = infos.X;
% end
return;


%% Step 3: Matching process 
%  this is only needed when np > 0 to match the existing anchors or
%  to see how well the found localizations for sensors match the their true
%  positions
%

t1 = tic;
if np > 0
  if (m == 0) %there exist no anchors
     PA =[];
     PS = PP;
     X1 = [];
     X2 = X(:, 1:np);
     [~, X2] = procrustes(PS', X2');
     X2 = X2';
  else        % there exist anchors and we must have (n > m)
     PA = PP(:, 1:m);
     X1 =  X(:, 1:m);
     [Q, X1, a0, p0] = procrustes_qi(PA, X1);
     
     X2 =  X(:, m+1:n); % number of known sensors
     X2 = X2 - p0(:, ones(n-m, 1));
     X2 = Q'*X2 + a0(:, ones(n-m, 1));      
  end

  X =[X1, X2];
  
  if (np > m)   % calculate RMSD only based on those known sensors
                % in most cases, np = n.
                % the code allow np < n (partial sensors known)
     PS = PP(:, (m+1):np);
     errtrue = sum((X2(:, 1:(np-m))-PS).^2);
     RMSD = sqrt(sum(errtrue))/sqrt(np);
  else
      RMSD = []; % PS =[]
  end
else
    RMSD = [];
end
fprintf('\n Time for Matching Process  ======= %.1f(secs) \n', toc(t1));

% Graphics before Refinement step
% markersize = 5;
% if (plotyes)
%    plotEMBED(X, dim, markersize, m, PP, RMSD);
% end

%% Step 4: Refinement step (code by Kim-Chuan)
%
%pars.Refinementyes

if ~isfield(pars, 'Refinementyes')
    pars.Refinementyes = 1; % take the refinement step
end
Refinement = pars.Refinementyes;

if (Refinement == 1) && (n0 <= m*(m-1)/2) % no fixed anchor-sensor distances
    fprintf('\n Running the gradient Refinement step ... \n');
    t1 = tic;
    [X2, ~] = refinepositions(X2, PA, DD);
    X = [X1, X2];
    if (np > m)
       RMSD = sum(sum((X2-PS).^2))/(np-m);
       RMSD = sqrt(RMSD); % calculate RMSD by Biswas et. al.
    end
    fprintf('\n Time for Refinement Step  ======= %.1f(secs) \n', toc(t1));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%% Step 5: Graphics (after Refinement step)
%
% plot in R^2 when PP is not empty
markersize = 5;
t1 = tic;

if (plotyes)
   plotEMBED(X, dim, markersize, m, PP, RMSD);
end
fprintf('\n Time for Graphics    =============== %.1f(secs) \n', toc(t1));
%%%%%%%% End of Graphics



% computing the final objective function value
infos.f = sqrt(sum(sum( (H.*(sqdistance(X)-Dold)).^2  )));

Time_used = toc(t0);

%% Output information
%
infos.t = Time_used;
infos.RMSD = RMSD;
infos.Y = Y; % computed EDM

% print out some key data
fprintf('\n Primal function value || Ho(Y-D)|| === %9.8e', full(infos.f));
fprintf('\n RMSD                               === %9.8e', RMSD);
fprintf('\n Total computing time               === %.1f(secs) \n', infos.t);

%%%%%% End of Main EMBED


function plotEMBED(X, dim, markersize, m, PP, RMSD)

if nargin < 3
    markersize = 5;
    m = 0;
    np = 0;
    PP = [];
end
if nargin < 4
    m = 0;
    np = 0;
end
if nargin >= 5
   np = size(PP, 2);
end

if nargin < 6
    RMSD = 0;
end

if (dim == 2)
 % plot the recovered anchors and sensors
  figure
 plot(X(1, :), X(2, :), '*r', 'markersize',markersize);
 hold on
 
 % plot the anchors
  if (m > 0) 
    h = plot(PP(1,1:m),PP(2,1:m),'d','markersize',markersize); 
    set(h,'linewidth',3,'color','b');
  end
 % plot the sensors
 if (np > m) 
     plot(PP(1, (m+1):end), PP(2, (m+1):end), 'ok', 'markersize',markersize); % plot the position of AS in R^2
 % link the known anchor/sensors and the recoverd points
     xy = [X(:, 1:np)'; PP'];
     I  = (1:np)'; J = (np+1:2*np)';
     a  = ones(np,1);
     E  = sparse(I,J,a, 2*np, 2*np);
     gplot(E, xy);
 end
     axis([-0.6 0.6 -0.6 0.6]);
   xlabel(['EMBED: RMSD = ', sprintf('%4.2e', RMSD)]);
end

% plot in R^3
% 
if (dim == 3)
    figure
    plot3(X(1, :), X(2, :), X(3, :), '*r', 'markersize',markersize);
    hold on; grid on;
   if (m > 0)
       h = plot3(PP(1,1:m),PP(2,1:m),PP(3,1:m),'d','markersize',markersize); 
       set(h,'linewidth',3,'color','b');     
   end  

    if (np > m)
       plot3(PP(1,(m+1):end),PP(2,(m+1):end),PP(3,(m+1):end),'ok','markersize',markersize);
       hold on; grid on;
       plot3([X(1,(m+1):end); PP(1,(m+1):end)],[X(2,(m+1):end); PP(2,(m+1):end)],[X(3,(m+1):end); PP(3,(m+1):end)],'b');
    end
     
      %axis('square'); 
      %axis(0.6*BoxScale*[-1,1,-1,1,-1,1])
      xlabel(['EMBED: RMSD = ', sprintf('%4.2e', RMSD)]);
     % pause(0.1);
      hold off
end


return
