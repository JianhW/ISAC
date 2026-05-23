function [z_opt, info] = Mine(z, idx_comm, idx_radar, varargin)
%MINE Joint PISL/PMEPR optimization for an OCDM frequency-domain waveform.
%
%   z_opt = Mine(z, idx_comm, idx_radar, 'lambda', lambda, ...)
%
% Inputs
%   z         : N-by-1 initial frequency-domain OCDM symbols.
%   idx_comm  : fixed communication indices, using MATLAB 1-based indexing.
%   idx_radar : radar indices to optimize, using MATLAB 1-based indexing.
%
% The function keeps z(idx_comm) unchanged and optimizes z(idx_radar) on
% the complex sphere with the same radar energy as the input. The returned
% vector has the same total energy as the input.
%
% Main options
%   'lambda'          : PMEPR weight in PISL + lambda * PMEPR_l objective.
%   'maxIt'           : maximum MM iterations.
%   'minIt'           : minimum iterations before convergence checks.
%   'lNormOrder'      : l in sum((N*|s_n|^2/E)^l).
%   'osFactor'        : oversampling factor used by the OCDM PAPR objective.
%   'epsStop'         : relative objective tolerance.
%   'lnormBound'      : 'current' or 'global'. Current is practical for
%                       high l and is protected by monotone backtracking.
%   'etaMode'         : 'exact' or 'upper'. Exact uses the 2N Gram matrix.
%   'maxBacktracking' : maximum monotone safeguard trials.
%   'backtrackingShrink' : shrink factor for the safeguard step.
%   'UpdateMode'      : Controls the update candidates used in each iteration.
%                       'fast'
%                           Fast recommended mode. Uses only the Riemannian
%                           tangent-gradient candidate on the radar-energy
%                           sphere plus monotone line search. It skips the
%                           expensive MM/Gram candidate and is usually much
%                           faster. This is the default.
%                       'hybrid'
%                           More accurate but slower mode. Tries the MM
%                           sphere-projection candidate, the direct gradient
%                           candidate, and the Riemannian tangent-gradient
%                           candidate, then accepts the one with the lowest
%                           true joint objective. Use this when final PAPR is
%                           more important than runtime.
%                       'mm'
%                           Pure MM-style sphere-projection candidate from the
%                           derivation. This is closest to the mathematical
%                           derivation, but can be conservative for l=100.
%                       'gradient'
%                           Direct negative-gradient candidate plus the
%                           Riemannian tangent-gradient candidate.
%   'mmPeriod'        : Optional compromise between fast and hybrid. If
%                       UpdateMode='fast' and mmPeriod>0, the function also
%                       evaluates the expensive MM candidate every mmPeriod
%                       iterations. Example: mmPeriod=20. Use 0 to disable.
%                       In current simulations, 'fast' with mmPeriod=0 is the
%                       best speed choice, while 'hybrid' is the strongest
%                       but slowest choice.
%   'verbose'         : print iteration diagnostics.
%
% Optional output info contains objective histories and convergence flags.

p = inputParser;
addRequired(p, 'z', @(x) isnumeric(x) && isvector(x));
addRequired(p, 'idx_comm', @(x) isnumeric(x) && isvector(x));
addRequired(p, 'idx_radar', @(x) isnumeric(x) && isvector(x));
addParameter(p, 'lambda', 1e-4, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'adaptiveLambda', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'targetPAPR', 2.0, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'targetPISL', 1e-5, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'lambdaMax', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 0));
addParameter(p, 'maxIt', 100, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'minIt', 20, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'lNormOrder', 8, @(x) isnumeric(x) && isscalar(x) && x >= 2);
addParameter(p, 'osFactor', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'epsStop', 1e-6, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'lnormBound', 'current', @(x) ischar(x) || isstring(x));
addParameter(p, 'etaMode', 'exact', @(x) ischar(x) || isstring(x));
addParameter(p, 'maxBacktracking', 12, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'backtrackingShrink', 0.5, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
addParameter(p, 'mmPeriod', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'tol', 1e-12, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'verbose', false, @(x) islogical(x) && isscalar(x));

addParameter(p, 'UpdateMode', 'fast', @(x) ischar(x) || isstring(x));
addParameter(p, 'PaprWeight', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));

parse(p, z, idx_comm, idx_radar, varargin{:});

lambda = p.Results.lambda;
baseLambda = lambda;
adaptiveLambda = p.Results.adaptiveLambda;
targetPAPR = p.Results.targetPAPR;
targetPISL = p.Results.targetPISL;
if isempty(p.Results.lambdaMax)
    lambdaMax = max(10 * baseLambda, baseLambda);
else
    lambdaMax = p.Results.lambdaMax;
end
maxIt = round(p.Results.maxIt);
minIt = min(round(p.Results.minIt), maxIt);
l_order = round(p.Results.lNormOrder);
osFactor = round(p.Results.osFactor);
epsStop = p.Results.epsStop;
lnormBound = lower(string(p.Results.lnormBound));
etaMode = lower(string(p.Results.etaMode));
updateMode = lower(string(p.Results.UpdateMode));
maxBacktracking = round(p.Results.maxBacktracking);
backtrackingShrink = p.Results.backtrackingShrink;
mmPeriod = round(p.Results.mmPeriod);
tol = p.Results.tol;
verbose = p.Results.verbose;

z = z(:);
idx_comm = idx_comm(:);
idx_radar = idx_radar(:);
N = numel(z);

validate_indices(idx_comm, idx_radar, N);

zc = z(idx_comm);
totalEnergy = real(sum(abs(z).^2));
commEnergy = real(sum(abs(zc).^2));
radarEnergy = totalEnergy - commEnergy;

if totalEnergy <= tol
    error('Mine:ZeroEnergy', 'Input z has zero energy.');
end
if radarEnergy < -1e-10
    error('Mine:InvalidEnergy', 'Communication energy exceeds the total input energy.');
end
radarEnergy = max(radarEnergy, 0);

if isempty(idx_radar) || radarEnergy <= tol
    z_opt = z;
    info = init_info(maxIt);
    M0 = osFactor * N;
    C0 = local_DFnT_adjoint_columns(M0, N);
    B0 = local_unitary_dft(N) * local_DFnT_adjoint_columns(N, N);
    pislScale0 = N / (totalEnergy^2);
    [info.objective(1), info.pisl(1), info.pmeprL(1), info.papr(1)] = ...
        objective_value(z_opt, B0, C0, M0, totalEnergy, lambda, l_order, pislScale0);
    info.iterations = 0;
    info.converged = true;
    info.status = 'no radar degrees of freedom';
    return;
end

% Enforce exact feasibility of the initial radar block while preserving the
% fixed communication symbols and input radar energy.
zr0 = z(idx_radar);
if norm(zr0) <= tol
    zr0 = randn(numel(idx_radar), 1) + 1j * randn(numel(idx_radar), 1);
end
z(idx_radar) = sqrt(radarEnergy) * zr0 / norm(zr0);
z(idx_comm) = zc;

% OCDM time transform used by the main script: time = Psi' * z.
% PAPR is evaluated on an oversampled OCDM signal, while PISL follows the
% N-point periodic-autocorrelation expression from the derivation.
M = osFactor * N;
C = local_DFnT_adjoint_columns(M, N);
CN = local_DFnT_adjoint_columns(N, N);
FN = local_unitary_dft(N);
B = FN * CN;

pislScale = N / (totalEnergy^2);

needsMM = any(updateMode == ["mm", "sphere", "hybrid"]) || mmPeriod > 0;
if needsMM
    BB_abs2 = abs(B * B').^2;
    BC_abs2 = abs(B * C').^2;
    CC_abs2 = abs(C * C').^2;
    normB2 = real(sum(abs(B).^2, 2));
    normC2 = real(sum(abs(C).^2, 2));
else
    BB_abs2 = [];
    BC_abs2 = [];
    CC_abs2 = [];
    normB2 = [];
    normC2 = [];
end

scalePower = M / totalEnergy;

info = init_info(maxIt);
[objPrev, pislPrev, pmeprPrev, paprPrev, pmeprRootPrev] = objective_value(z, B, C, M, totalEnergy, lambda, l_order, pislScale);
info.objective(1) = objPrev;
info.pisl(1) = pislPrev;
info.pmeprL(1) = pmeprPrev;
info.papr(1) = paprPrev;
info.pmeprRoot(1) = pmeprRootPrev;

for it = 1:maxIt
    if adaptiveLambda && it > 1
        if paprPrev > targetPAPR
            ratio = min(10, max(1, paprPrev / targetPAPR));
            lambda = min(lambdaMax, max(baseLambda, baseLambda * ratio^2));
        elseif pislPrev > targetPISL
            lambda = max(baseLambda, 0.7 * lambda);
        else
            lambda = baseLambda;
        end
    end

    s_time = C * z;
    x = real(abs(s_time).^2);
    pwr = scalePower * x;
    pwrPeakScale = max(max(pwr), 1);
    pwrStable = pwr / pwrPeakScale;
    pmeprStableSum = sum(max(pwrStable, 0).^l_order);
    if pmeprStableSum <= realmin
        rootGradScale = 0;
    else
        rootGradScale = pwrPeakScale * (1 / l_order) * pmeprStableSum^(1 / l_order - 1);
    end

    [alphaX, betaX, deltaX] = pmepr_quadratic_params(pwrStable, scalePower / pwrPeakScale, ...
        l_order, lnormBound, N / pwrPeakScale);

    d = real(abs(B * z).^2);
    v = rootGradScale * (scalePower / pwrPeakScale) * l_order * max(pwrStable, 0).^(l_order - 1);

    HB = 2 * pislScale * (B' * bsxfun(@times, d, B));
    HC = lambda * (C' * bsxfun(@times, v, C));
    grad = (HB + HC) * z;

    eta = 0;
    candidates = {};

    useMMThisIter = updateMode == "mm" || updateMode == "sphere" || updateMode == "hybrid" || ...
        (mmPeriod > 0 && mod(it, mmPeriod) == 0);

    if useMMThisIter
        eta = compute_eta(alphaX, deltaX, lambda, pislScale, etaMode, ...
            BB_abs2, BC_abs2, CC_abs2, normB2, normC2, N);

        H = HB + HC - 2 * eta * (z * z');
        H = (H + H') / 2;

        muMax = max(d);
        gammaMax = max(v);
        UScalar = 2 * muMax + lambda * gammaMax;

        upsilonMM = UScalar * z - H * z;
        zMM = project_candidate(z, zc, idx_comm, idx_radar, upsilonMM, radarEnergy, tol);
        candidates{end + 1} = evaluate_candidate('mm', z, zMM, zc, idx_comm, idx_radar, ...
            radarEnergy, objPrev, B, C, M, totalEnergy, lambda, l_order, pislScale, ...
            maxBacktracking, backtrackingShrink, tol);
    end

    if updateMode == "gradient" || updateMode == "hybrid"
        upsilonGrad = -grad;
        zGrad = project_candidate(z, zc, idx_comm, idx_radar, upsilonGrad, radarEnergy, tol);
        candidates{end + 1} = evaluate_candidate('gradient', z, zGrad, zc, idx_comm, idx_radar, ...
            radarEnergy, objPrev, B, C, M, totalEnergy, lambda, l_order, pislScale, ...
            maxBacktracking, backtrackingShrink, tol);
    end

    if updateMode == "fast" || updateMode == "gradient" || updateMode == "hybrid"
        candidates{end + 1} = riemannian_candidate('riemannian', z, grad, zc, idx_comm, idx_radar, ...
            radarEnergy, objPrev, B, C, M, totalEnergy, lambda, l_order, pislScale, ...
            maxBacktracking, backtrackingShrink, tol);
    end

    if isempty(candidates)
        error('Mine:BadUpdateMode', 'UpdateMode must be ''fast'', ''mm'', ''gradient'', or ''hybrid''.');
    end

    bestIdx = 1;
    for ci = 2:numel(candidates)
        if candidates{ci}.obj < candidates{bestIdx}.obj
            bestIdx = ci;
        end
    end

    best = candidates{bestIdx};
    zCandidate = best.z;
    objCandidate = best.obj;
    pislCandidate = best.pisl;
    pmeprCandidate = best.pmepr;
    paprCandidate = best.papr;
    acceptedStep = best.step;
    usedBacktracking = best.usedBacktracking;
    source = best.source;

    relChange = abs(objPrev - objCandidate) / max(1, abs(objPrev));
    updateNorm = norm(zCandidate - z) / max(norm(z), tol);

    z = zCandidate;
    objPrev = objCandidate;
    paprPrev = paprCandidate;
    pmeprRootPrev = best.pmeprRoot;
    pislPrev = pislCandidate;

    info.objective(it + 1) = objCandidate;
    info.pisl(it + 1) = pislCandidate;
    info.pmeprL(it + 1) = pmeprCandidate;
    info.papr(it + 1) = paprCandidate;
    info.pmeprRoot(it + 1) = best.pmeprRoot;
    info.eta(it) = eta;
    info.lambda(it) = lambda;
    info.acceptedStep(it) = acceptedStep;
    info.usedBacktracking(it) = usedBacktracking;
    info.updateNorm(it) = updateNorm;
    info.gradNorm(it) = projected_grad_norm(grad, z, idx_radar, radarEnergy, tol);
    info.source{it} = source;

    if verbose
        fprintf(['[Mine] it=%3d obj=%.6e PISL=%.6e PAPR=%.3f dB ', ...
            'rel=%.3e step=%.3g eta=%.3e src=%s\n'], ...
            it, objCandidate, pislCandidate, 10 * log10(paprCandidate), ...
            relChange, acceptedStep, eta, source);
    end

    if it >= minIt && (relChange <= epsStop || updateNorm <= tol)
        info.converged = true;
        info.status = 'relative objective tolerance reached';
        break;
    end
end

info.iterations = find(info.objective > -inf, 1, 'last') - 1;
if ~info.converged
    info.status = 'maximum iterations reached';
end

fields = {'objective', 'pisl', 'pmeprL', 'pmeprRoot', 'papr'};
for k = 1:numel(fields)
    name = fields{k};
    info.(name) = info.(name)(1:info.iterations + 1);
end
fields = {'eta', 'acceptedStep', 'usedBacktracking', 'updateNorm', 'gradNorm'};
for k = 1:numel(fields)
    name = fields{k};
    info.(name) = info.(name)(1:max(info.iterations, 0));
end
info.source = info.source(1:max(info.iterations, 0));
info.lambda = info.lambda(1:max(info.iterations, 0));

z(idx_comm) = zc;
z(idx_radar) = sqrt(radarEnergy) * z(idx_radar) / max(norm(z(idx_radar)), tol);
z_opt = z;

end

function info = init_info(maxIt)
info = struct();
info.objective = -inf(maxIt + 1, 1);
info.pisl = -inf(maxIt + 1, 1);
info.pmeprL = -inf(maxIt + 1, 1);
info.pmeprRoot = -inf(maxIt + 1, 1);
info.papr = -inf(maxIt + 1, 1);
info.eta = zeros(maxIt, 1);
info.lambda = zeros(maxIt, 1);
info.acceptedStep = zeros(maxIt, 1);
info.usedBacktracking = false(maxIt, 1);
info.updateNorm = zeros(maxIt, 1);
info.gradNorm = zeros(maxIt, 1);
info.source = cell(maxIt, 1);
info.iterations = 0;
info.converged = false;
info.status = '';
end

function validate_indices(idx_comm, idx_radar, N)
if any(idx_comm < 1) || any(idx_comm > N) || any(idx_comm ~= round(idx_comm))
    error('Mine:BadIndex', 'idx_comm must contain integer indices in [1, N].');
end
if any(idx_radar < 1) || any(idx_radar > N) || any(idx_radar ~= round(idx_radar))
    error('Mine:BadIndex', 'idx_radar must contain integer indices in [1, N].');
end
if ~isempty(intersect(idx_comm(:), idx_radar(:)))
    error('Mine:OverlapIndex', 'idx_comm and idx_radar must be disjoint.');
end
end

function F = local_unitary_dft(N)
n = (0:N-1).';
k = 0:N-1;
F = exp(-1j * 2 * pi / N * (n * k)) / sqrt(N);
end

function Psi = local_DFnT(P)
m = (0:P-1)' * ones(1, P);
n = ones(P, 1) * (0:P-1);
fresnelKernel = exp(1j * pi / P * (m - n).^2);
normalization = (1 / sqrt(P)) * exp(1j * pi / 4);
Psi = normalization * fresnelKernel;
end

function C = local_DFnT_adjoint_columns(P, numCols)
rowIdx = (0:P-1).';
colIdx = 0:(numCols - 1);
normalization = (1 / sqrt(P)) * exp(1j * pi / 4);
C = conj(normalization) * exp(-1j * pi / P * (rowIdx - colIdx).^2);
end

function [obj, pisl, pmeprL, paprValue, pmeprRoot] = objective_value(z, B, C, N, totalEnergy, lambda, l_order, pislScale)
s_time = C * z;
pwr = (N / totalEnergy) * real(abs(s_time).^2);
spec = B * z;

pisl = pislScale * sum(real(abs(spec).^4)) - 1;
pisl = max(real(pisl), 0);
paprValue = max(pwr);
pScale = max(paprValue, 1);
stablePower = max(pwr, 0) / pScale;
pmeprLStable = sum(stablePower.^l_order);
pmeprRoot = pScale * pmeprLStable^(1 / l_order);
pmeprL = pmeprLStable * pScale^l_order;
obj = pisl + lambda * pmeprRoot;
end

function [alphaX, betaX, deltaX] = pmepr_quadratic_params(pwr, scalePower, l_order, lnormBound, activeDim)
switch lnormBound
    case "global"
        t = activeDim;
    case "current"
        t = max(max(pwr), 1);
    otherwise
        error('Mine:BadBound', 'lnormBound must be ''current'' or ''global''.');
end

y0 = max(real(pwr(:)), 0);
alphaY = zeros(size(y0));
betaY = zeros(size(y0));

nearTop = abs(t - y0) <= 1e-10 * max(1, t);
regular = ~nearTop;

if any(regular)
    yr = y0(regular);
    alphaY(regular) = (t^l_order - yr.^l_order - l_order * yr.^(l_order - 1) .* (t - yr)) ./ (t - yr).^2;
    betaY(regular) = l_order * yr.^(l_order - 1) - 2 * alphaY(regular) .* yr;
end

if any(nearTop)
    yt = max(t, 0);
    alphaY(nearTop) = 0.5 * l_order * (l_order - 1) * yt^(l_order - 2);
    betaY(nearTop) = l_order * yt^(l_order - 1) - 2 * alphaY(nearTop) * yt;
end

alphaY = max(real(alphaY), 0);
alphaX = alphaY * scalePower^2;
betaX = betaY * scalePower;

tiny = alphaX <= eps(max(1, max(alphaX)));
deltaX = zeros(size(alphaX));
deltaX(~tiny) = betaX(~tiny) ./ (2 * alphaX(~tiny));
end

function eta = compute_eta(alphaX, deltaX, lambda, pislScale, etaMode, BB_abs2, BC_abs2, CC_abs2, normB2, normC2, N)
switch etaMode
    case {"exact", "gram"}
        sqrtAlpha = sqrt(max(alphaX(:), 0));
        sqrtLambdaAlpha = sqrt(max(lambda, 0)) * sqrtAlpha;

        Kqq = pislScale * BB_abs2;
        Kqm = bsxfun(@times, BC_abs2 + normB2(:) * deltaX(:).', sqrtLambdaAlpha(:).');

        mmInner = CC_abs2 ...
            + normC2(:) * deltaX(:).' ...
            + deltaX(:) * normC2(:).' ...
            + N * (deltaX(:) * deltaX(:).');
        Kmm = lambda * (sqrtAlpha(:) * sqrtAlpha(:).') .* mmInner;

        K = [Kqq, Kqm; Kqm', Kmm];
        K = (K + K') / 2;
        ev = eig(K, 'vector');
        eta = max(real(ev));

    case {"upper", "bound"}
        etaPhi = pislScale;
        sqrtAlpha = sqrt(max(alphaX(:), 0));
        mmInner = CC_abs2 ...
            + normC2(:) * deltaX(:).' ...
            + deltaX(:) * normC2(:).' ...
            + N * (deltaX(:) * deltaX(:).');
        gammaGram = (sqrtAlpha(:) * sqrtAlpha(:).') .* mmInner;
        gammaGram = (gammaGram + gammaGram') / 2;
        eta = etaPhi + lambda * max(real(eig(gammaGram, 'vector')));

    otherwise
        error('Mine:BadEtaMode', 'etaMode must be ''exact'' or ''upper''.');
end

eta = max(real(eta), 0);
end

function zCandidate = project_candidate(z, zc, idx_comm, idx_radar, upsilon, radarEnergy, tol)
zCandidate = z;
zCandidate(idx_comm) = zc;
u = upsilon(idx_radar);
if norm(u) <= tol || any(~isfinite(u))
    return;
end
zCandidate(idx_radar) = sqrt(radarEnergy) * u / norm(u);
end

function out = evaluate_candidate(source, zOld, zFullStep, zc, idx_comm, idx_radar, ...
    radarEnergy, objOld, B, C, N, totalEnergy, lambda, l_order, pislScale, maxBacktracking, shrink, tol)

[objFull, pislFull, pmeprFull, paprFull, pmeprRootFull] = objective_value(zFullStep, B, C, N, totalEnergy, lambda, l_order, pislScale);
out = struct();
out.source = source;
out.z = zFullStep;
out.obj = objFull;
out.pisl = pislFull;
out.pmepr = pmeprFull;
out.papr = paprFull;
out.pmeprRoot = pmeprRootFull;
out.step = 1;
out.usedBacktracking = false;

if objFull <= objOld * (1 + 10 * eps) + 1e-14
    return;
end

[zBest, objBest, pislBest, pmeprBest, paprBest, pmeprRootBest, acceptedStep] = ...
    monotone_backtracking(zOld, zFullStep, zc, idx_comm, idx_radar, ...
    radarEnergy, objOld, B, C, N, totalEnergy, lambda, l_order, pislScale, ...
    maxBacktracking, shrink, tol);

out.z = zBest;
out.obj = objBest;
out.pisl = pislBest;
out.pmepr = pmeprBest;
out.papr = paprBest;
out.pmeprRoot = pmeprRootBest;
out.step = acceptedStep;
out.usedBacktracking = true;
end

function out = riemannian_candidate(source, zOld, grad, zc, idx_comm, idx_radar, ...
    radarEnergy, objOld, B, C, N, totalEnergy, lambda, l_order, pislScale, maxBacktracking, shrink, tol)

out = struct();
out.source = source;
out.z = zOld;
[out.obj, out.pisl, out.pmepr, out.papr, out.pmeprRoot] = ...
    objective_value(zOld, B, C, N, totalEnergy, lambda, l_order, pislScale);
out.step = 0;
out.usedBacktracking = true;

if isempty(idx_radar) || radarEnergy <= tol
    return;
end

r = zOld(idx_radar);
g = grad(idx_radar);
if norm(r) <= tol || norm(g) <= tol
    return;
end

radialCoeff = real(r' * g) / max(real(r' * r), tol);
gProj = g - radialCoeff * r;
if norm(gProj) <= tol || any(~isfinite(gProj))
    return;
end

baseStep = sqrt(radarEnergy) / max(norm(gProj), tol);
step = baseStep;
for bt = 1:maxBacktracking
    rTrial = r - step * gProj;
    if norm(rTrial) <= tol
        step = step * shrink;
        continue;
    end

    zTrial = zOld;
    zTrial(idx_comm) = zc;
    zTrial(idx_radar) = sqrt(radarEnergy) * rTrial / norm(rTrial);

    [objTrial, pislTrial, pmeprTrial, paprTrial, pmeprRootTrial] = ...
        objective_value(zTrial, B, C, N, totalEnergy, lambda, l_order, pislScale);

    if objTrial <= objOld * (1 + 10 * eps) + 1e-14
        out.z = zTrial;
        out.obj = objTrial;
        out.pisl = pislTrial;
        out.pmepr = pmeprTrial;
        out.papr = paprTrial;
        out.pmeprRoot = pmeprRootTrial;
        out.step = step / baseStep;
        return;
    end

    step = step * shrink;
end
end

function gnorm = projected_grad_norm(grad, z, idx_radar, radarEnergy, tol)
if radarEnergy <= tol || isempty(idx_radar)
    gnorm = 0;
    return;
end
r = z(idx_radar);
g = grad(idx_radar);
if norm(r) <= tol
    gnorm = norm(g);
    return;
end
radialCoeff = real(r' * g) / max(real(r' * r), tol);
gProj = g - radialCoeff * r;
gnorm = norm(gProj) / max(norm(g), tol);
end

function [zBest, objBest, pislBest, pmeprBest, paprBest, pmeprRootBest, acceptedStep] = ...
    monotone_backtracking(zOld, zFullStep, zc, idx_comm, idx_radar, radarEnergy, ...
    objOld, B, C, N, totalEnergy, lambda, l_order, pislScale, maxBacktracking, shrink, tol)

zBest = zOld;
[objBest, pislBest, pmeprBest, paprBest, pmeprRootBest] = objective_value(zOld, B, C, N, totalEnergy, lambda, l_order, pislScale);
acceptedStep = 0;

rOld = zOld(idx_radar);
rFull = zFullStep(idx_radar);

step = 1;
for bt = 1:maxBacktracking
    rTrial = (1 - step) * rOld + step * rFull;
    if norm(rTrial) <= tol
        step = step * shrink;
        continue;
    end

    zTrial = zOld;
    zTrial(idx_comm) = zc;
    zTrial(idx_radar) = sqrt(radarEnergy) * rTrial / norm(rTrial);

    [objTrial, pislTrial, pmeprTrial, paprTrial, pmeprRootTrial] = ...
        objective_value(zTrial, B, C, N, totalEnergy, lambda, l_order, pislScale);
    if objTrial <= objOld * (1 + 10 * eps) + 1e-14
        zBest = zTrial;
        objBest = objTrial;
        pislBest = pislTrial;
        pmeprBest = pmeprTrial;
        paprBest = paprTrial;
        pmeprRootBest = pmeprRootTrial;
        acceptedStep = step;
        return;
    end

    step = step * shrink;
end
end
