%% Verify convergence of Mine.m for the OCDM joint PISL/PMEPR algorithm.
% This script is independent from OCDM_CCDF_main.m. It generates feasible
% frequency-domain symbols, runs Mine.m, checks monotonic objective descent,
% and plots convergence diagnostics.

clear; clc; close all;
rng(7);

%% Test parameters
N = 128;
Nc = 32;
e = 0.1;
lambda = 1.6e-3;
lNormOrder = 100;
maxIt = 1000;
minIt = 200;
epsStop = 1e-7;
nTrials = 3;
osFactor = 4;

objectiveTol = 1e-9;
commTol = 1e-10;
energyTol = 1e-10;
targetPaprLinear = 2.0;
targetPisl = 1e-5;

allPassed = true;
trialInfo = cell(nTrials, 1);

fprintf('Verifying Mine.m convergence with N=%d, Nc=%d, lambda=%g, l=%d\n', ...
    N, Nc, lambda, lNormOrder);

for trial = 1:nTrials
    %% Generate one feasible waveform
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

    Er = 1 - e;
    zr = randn(Nr, 1) + 1j * randn(Nr, 1);
    zr = sqrt(Er) * zr / norm(zr);

    z0 = zeros(N, 1);
    z0(idx_comm) = zc;
    z0(idx_radar) = zr;

    %% Run the proposed algorithm
    [zOpt, info] = Mine(z0, idx_comm, idx_radar, ...
        'lambda', lambda, ...
        'adaptiveLambda', false, ...
        'targetPAPR', targetPaprLinear, ...
        'targetPISL', targetPisl, ...
        'lambdaMax', 1e-2, ...
        'osFactor', osFactor, ...
        'maxIt', maxIt, ...
        'minIt', minIt, ...
        'lNormOrder', lNormOrder, ...
        'epsStop', epsStop, ...
        'lnormBound', 'global', ...
        'etaMode', 'exact', ...
        'maxBacktracking', 16, ...
        'backtrackingShrink', 0.5, ...
        'UpdateMode', 'hybrid', ...
        'verbose', false);

    trialInfo{trial} = info;

    %% Feasibility checks
    commErr = norm(zOpt(idx_comm) - z0(idx_comm));
    radarEnergyErr = abs(sum(abs(zOpt(idx_radar)).^2) - Er);
    totalEnergyErr = abs(sum(abs(zOpt).^2) - 1);

    objDiff = diff(info.objective(:));
    pislDiff = diff(info.pisl(:));
    paprDiff = diff(info.papr(:));
    maxIncrease = max([0; objDiff]);
    maxPislIncrease = max([0; pislDiff]);
    maxPaprIncrease = max([0; paprDiff]);
    isMonotone = maxIncrease <= objectiveTol * max(1, abs(info.objective(1)));
    pislMostlyDescends = info.pisl(end) <= info.pisl(1);
    paprMostlyDescends = info.papr(end) <= info.papr(1);

    improved = info.objective(end) <= info.objective(1) + objectiveTol * max(1, abs(info.objective(1)));
    feasible = commErr <= commTol && radarEnergyErr <= energyTol && totalEnergyErr <= energyTol;
    tailRel = abs(info.objective(end) - info.objective(max(1, end - 1))) / ...
        max(1, abs(info.objective(max(1, end - 1))));
    convergedByCriterion = info.converged || tailRel <= 1e-6;
    targetReached = info.papr(end) <= targetPaprLinear && info.pisl(end) <= targetPisl;
    passed = isMonotone && improved && feasible && convergedByCriterion && ...
        pislMostlyDescends && paprMostlyDescends && targetReached && all(isfinite(info.objective));

    allPassed = allPassed && passed;

    fprintf(['Trial %d/%d: %s | iter=%d | obj %.6e -> %.6e | ', ...
        'PISL %.3e -> %.3e | PAPR %.3f -> %.3f dB | ', ...
        'maxObjInc %.3e | maxPislInc %.3e | maxPaprInc %.3e | tailRel %.3e | target=%d | commErr %.1e | ErErr %.1e\n'], ...
        trial, nTrials, passfail(passed), info.iterations, ...
        info.objective(1), info.objective(end), ...
        info.pisl(1), info.pisl(end), ...
        10 * log10(info.papr(1)), 10 * log10(info.papr(end)), ...
        maxIncrease, maxPislIncrease, maxPaprIncrease, tailRel, targetReached, commErr, radarEnergyErr);

    if ~passed
        fprintf('  Status: %s\n', info.status);
    end
end

%% Plot convergence curves for the first trial
info = trialInfo{1};
figure('Color', 'w', 'Name', 'Mine convergence verification');

subplot(3, 1, 1);
semilogy(0:info.iterations, info.objective, 'LineWidth', 1.8);
grid on;
xlabel('Iteration');
ylabel('Objective');
title('Joint objective');

subplot(3, 1, 2);
plot(0:info.iterations, 10 * log10(info.papr), 'LineWidth', 1.8);
grid on;
xlabel('Iteration');
ylabel('PAPR (dB)');
title('OCDM time-domain PAPR');

subplot(3, 1, 3);
semilogy(0:info.iterations, max(info.pisl, eps), 'LineWidth', 1.8);
grid on;
xlabel('Iteration');
ylabel('Normalized PISL');
title('Normalized PISL');

if allPassed
    fprintf('All convergence checks passed.\n');
else
    error(['Mine convergence verification failed. Check the mathematical ', ...
        'majorization steps, the lambda/l scaling, and the implementation.']);
end

function s = passfail(flag)
if flag
    s = 'PASS';
else
    s = 'FAIL';
end
end
