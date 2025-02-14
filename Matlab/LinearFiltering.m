function [Y] = LinearFiltering(X)
a = size(X);
N = a(1, 1);
Y = X;
Y(1, :) = (X(1, :) + X(2, :)) ./ 2;
Y(N, :) = (X(N, :) + X(N-1, :)) ./ 2;
for i = 2:(N - 1)
    Y(i, :) = (X(i, :) + X(i+1, :) + X(i-1, :)) ./ 3;
end;
return;
