%%% Joint Modeled-Based Distance Embedding
%%% Contact: Yuping Zhang (yuping.zhang@uconn.edu) and Zhengqing Ouyang ( ouyang@schoolph.umass.edu ) 
%%% track-specific beta
function [finalY, finalbeta, finalbeta0, y, beta0vec, finalinfo, allY, value] = JwMDE(varargin)

% Input in varargin:
%   1: Number of tracks (M)
%   2: K \times M matrix for beta grids for track 1,...M; each column is a vector of beta values
%   for one track
%   3 to M+2: M contact maps
%   
% Output:
%   finalY: final N * 3 structure
%   finalbeta: selected beta
%   finalbeta0: selected beta0
%   y: log-likelihood, first row is overall EDM constrained, remain rows
%   for each track
%   beta0: estimated beta0s for beta grids
%   finalinfo: EDM embedding info of the selected beta
%   allY: all structures under beta grid
%   value: stress values under beta grid

M = varargin{1};

if(M>1)
    betavec = varargin{2};
    Nbeta = size(betavec, 1); 
    sizeN = size(varargin{3}); sizeN = sizeN(1);
    contactMap = zeros(sizeN, sizeN, M);
    for i = 1:M
        % first make sure contact map is symmetric
        contactMap(:, :, i) = (varargin{i + 2} + transpose(varargin{i + 2})) ./ 2;
    end
elseif(M==1)  %% M=1, Modeled-Based Distance Embedding (MDE)
    betavec = varargin{2};
    Nbeta = length(betavec);
    sizeN = size(varargin{3}); sizeN = sizeN(1);
    contactMap = zeros(sizeN, sizeN);
    contactMap = (varargin{3} + transpose(varargin{3})) ./ 2;  %%% only one track
end

% Preapre EMBED.m parameters
pars.plotyes = 0;
pars.m = 0;
pars.anchorconstraintyes = 0;
pars.spathyes = 0;
pars.refinement = 0;
pars.weight = 1;

% pars.H -- H weights (default: H =spones(D)) defined by EMBED.m
% pars.pseudocount = 1;
dim = 3;

% Grid search for optimal beta
if(M>1)
    beta0vec= betavec;
    y = zeros(M+1, Nbeta);
    value = zeros(1, Nbeta);
    allY = zeros(sizeN, 3, Nbeta);
    mybetas = betavec(1,:);
    for k = 1:Nbeta
        mybetas = betavec(k,:);
        Data=contactMap;
        for m = 1:M
            data = contactMap(:,:,m);
            data = 1 ./ (data+1);  % add pseudo ones; check those zeros 
            Data(:,:,m) = data .^ (2. / mybetas(m)); % updated pre-distance matrices, Data contains transformed contact mapps (data_ij = contact_map_ij^{-2/beta});
        end
        Dtemp = mean(Data, 3);
        Dtemp(logical(eye(sizeN))) = diag(0); % set the diagnal values as 0s.  
        [Y, infos] = EMBED(Dtemp, dim, pars); % EDM embedding
        value(1, k) = infos.f; % stress value
        Dsqpredict = sqdistance(Y); % squared pairwise distance
     
        Y = transpose(Y);  
        negloglikelihood=0;
        for m = 1:M
            tmp = contactMap(:, :, m);
            [negloglikelihoodm, beta0] = NegPoiLogLikelihood(tmp(tril(true(size(tmp)), -1)), Dsqpredict(tril(true(size(tmp)), -1)), -mybetas(m)/2); % evaluate negative loglikelihood
            negloglikelihood = negloglikelihood + negloglikelihoodm;
            y(m+1,k) = negloglikelihoodm; %%% each track likelihood
            beta0vec(k,m) = beta0;
        end
        Ysmooth = LinearFiltering(Y); % linear filtering
        allY(:, :, k) =  Ysmooth;
    
        y(1, k) = negloglikelihood;   %%% overall likelihood
   
        if(k==1)
             finalnegloglikelihood = negloglikelihood;
             finalbeta = mybetas(:);
             finalbeta0 = beta0vec(k,:);
             finalY = allY(:, :, k);
             finalinfo = infos;     
        elseif(negloglikelihood < finalnegloglikelihood)
            finalnegloglikelihood = negloglikelihood;
            finalbeta = mybetas(:);
            finalbeta0 = beta0vec(k,:);
            finalY = allY(:, :, k);
            finalinfo = infos;
        end
    end
elseif(M==1) %% M=1, Modeled-Based Distance Embedding (MDE)
    beta0vec= betavec;
    y = zeros(1, Nbeta);
    value = zeros(1, Nbeta);
    allY = zeros(sizeN, 3, Nbeta);
    mybetas = betavec(1);
    for k = 1:Nbeta
        mybetas = betavec(k);
        Data = contactMap;
        Data = 1 ./ (Data+1); % Data+1 avoid zero counts (XL comment)
        Data = Data .^ (2. / mybetas); % updated pre-distance matrices, Data contains transformed contact mapps (data_ij = contact_map_ij^{-2/beta});
        Dtemp = Data;
        Dtemp(logical(eye(sizeN))) = diag(0); % set the diagnal values as 0s. 
       
        %H = 1 ./ Dtemp;
        %H(logical(eye(sizeN))) = diag(0);
        %H(contactMap==0) = 0 ;
        H = spones(contactMap);
        pars.H = H;
        
        [Y, infos] = EMBED(Dtemp, dim, pars); % EDM embedding
        value(1, k) = infos.f; % stress value
        Dsqpredict = sqdistance(Y); % squared pairwise distance
     
        Y = transpose(Y);
     
        negloglikelihood=0;
        tmp = contactMap;
        [negloglikelihoodm, beta0] = NegPoiLogLikelihood(tmp(tril(true(size(tmp)), -1)), Dsqpredict(tril(true(size(tmp)), -1)), -mybetas/2);
        negloglikelihood=  negloglikelihoodm;
        beta0vec(k) = beta0;
        Ysmooth = LinearFiltering(Y); % linear filtering
        allY(:, :, k) =  Ysmooth;
      
        y(1, k) = negloglikelihood;   %%% overall likelihood
     
        if(k==1)
             finalnegloglikelihood = negloglikelihood;
             finalbeta = mybetas;
             finalbeta0 = beta0vec(k);
             finalY = allY(:, :, k);
             finalinfo = infos;     
        elseif(negloglikelihood < finalnegloglikelihood)
            finalnegloglikelihood = negloglikelihood;
            finalbeta = mybetas;
            finalbeta0 = beta0vec(k);
            finalY = allY(:, :, k);
            finalinfo = infos;
        end
    end
end
return;

