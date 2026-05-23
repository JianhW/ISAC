%% Diagnose lambda tradeoff for Mine.m.
clear; clc; close all;
rng(11);

N = 128;
Nc = 32;
e = 0.1;
osFactor = 4;
lNormOrder = 100;
maxIt = 1000;
minIt = 200;
epsStop = 1e-7;

lambdaList = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1];

rand_idx = randperm(N);
idx_comm = sort(rand_idx(1:Nc)).';
mask_comm = false(N, 1);
mask_comm(idx_comm) = true;
idx_radar = find(~mask_comm);
Nr = numel(idx_radar);

bits = randi([0 1], 2 * Nc, 1);
zc_unit = (1 / sqrt(2)) * ...
    ((2 * bits(1:2:end) - 1) + 1j * (2 * bits(2:2:end) - 1));
zc = sqrt(e) * zc_unit / sqrt(Nc);

zr = randn(Nr, 1) + 1j * randn(Nr, 1);
zr = sqrt(1 - e) * zr / norm(zr);

z0 = zeros(N, 1);
z0(idx_comm) = zc;
z0(idx_radar) = zr;

fprintf('lambda sweep, N=%d, Nc=%d, l=%d, os=%d\n', N, Nc, lNormOrder, osFactor);
fprintf('%12s %8s %14s %14s %12s %12s %12s\n', ...
    'lambda', 'iter', 'obj_end', 'PISL_end', 'PAPR_lin', 'PAPR_dB', 'conv');

result = zeros(numel(lambdaList), 6);
for k = 1:numel(lambdaList)
    lambda = lambdaList(k);
    [~, info] = Mine(z0, idx_comm, idx_radar, ...
        'lambda', lambda, ...
        'osFactor', osFactor, ...
        'maxIt', maxIt, ...
        'minIt', minIt, ...
        'lNormOrder', lNormOrder, ...
        'epsStop', epsStop, ...
        'lnormBound', 'current', ...
        'etaMode', 'upper', ...
        'UpdateMode', 'hybrid', ...
        'maxBacktracking', 20, ...
        'backtrackingShrink', 0.5, ...
        'verbose', false);

    result(k, :) = [lambda, info.iterations, info.objective(end), ...
        info.pisl(end), info.papr(end), info.converged];
    fprintf('%12.3g %8d %14.6e %14.6e %12.4f %12.3f %12d\n', ...
        lambda, info.iterations, info.objective(end), info.pisl(end), ...
        info.papr(end), 10 * log10(info.papr(end)), info.converged);
end

figure('Color', 'w', 'Name', 'Mine lambda sweep');
yyaxis left;
loglog(result(:, 1), result(:, 4), '-o', 'LineWidth', 1.6);
ylabel('Normalized PISL');
yyaxis right;
semilogx(result(:, 1), result(:, 5), '-s', 'LineWidth', 1.6);
ylabel('PAPR (linear)');
grid on;
xlabel('\lambda');
title('PISL/PAPR tradeoff');
