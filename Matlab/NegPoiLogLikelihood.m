%%%% Negative log-likelihood

function [negloglikelihood, beta0] = NegPoiLogLikelihood(vecN, vecd, beta)
    vecd(find(vecd == 0)) = 1e-200;

    beta0 = log(sum(vecN)) - log(sum(vecd .^ beta));  % beta was inputted as -mybetas(m)/2
    mu = beta0 + beta .* log(vecd);
    loglikelihood = sum(vecN .* mu - exp(mu)) - sum(gammaln(vecN + 1));
    negloglikelihood = -loglikelihood;
return;