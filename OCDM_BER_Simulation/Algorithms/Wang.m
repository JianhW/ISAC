function C_opt = Wang(c, C_idx, R_idx, varargin)
%WANG Optimized iterative clipping/filtering comparison algorithm.
%   C_opt = Wang(c, C_idx, R_idx) keeps the simulation interface and returns
%   an OFDM-domain codeword for the existing time_signal = FH(C_opt) call in
%   SIM_PISL_main.m.
%
%   This implementation follows the Wang/Luo-style optimized iterative
%   clipping and filtering code in:
%       E:\MyDocuments\ISAC\simulation\对比文献\对比文献2\OFDM\cpt11_10_1.m
%
%   The important fix for the ISL simulation is that R_idx is validated only
%   for interface compatibility. The CVX EVM constraint is applied to the
%   full input codeword, as in the comparison source. The previous port used
%   only C_idx and also added H(R_idx).*c_in(R_idx)=0; both choices let the
%   radar tones drift or vanish and destroy the expected sidelobe reduction.
%
%   Name-value options:
%       'L'       : oversampling factor, default 4
%       'Iter'    : number of optimized ICF iterations, default 2
%       'CR'      : clipping ratio, default sqrt(10^(5/10))
%       'UseCVX'  : use the CVX projection when available, default true
%       'Verbose' : print diagnostics, default false

if nargin < 3
    error('Wang:NotEnoughInputs', ...
        'Wang requires c, C_idx, and R_idx inputs.');
end

c = c(:);
C_idx = C_idx(:);
R_idx = R_idx(:);
N = numel(c);

validate_indices(C_idx, R_idx, N);

p = inputParser;
addParameter(p, 'L', 4, @(x) isscalar(x) && x >= 1 && mod(x, 1) == 0);
addParameter(p, 'Iter', 2, @(x) isscalar(x) && x >= 1 && mod(x, 1) == 0);
addParameter(p, 'CR', sqrt(10^(5/10)), @(x) isscalar(x) && x > 0);
addParameter(p, 'UseCVX', true, @islogical);
addParameter(p, 'Verbose', false, @islogical);
parse(p, varargin{:});

L = p.Results.L;
iters = p.Results.Iter;
CR = p.Results.CR;
useCVX = p.Results.UseCVX;
verbose = p.Results.Verbose;
LN = L * N;

A = build_ofdm_synthesis_matrix(N, L);
x = ofdm_synth(c, L);

for it = 1:iters
    Tclip = CR * norm(x) / sqrt(LN);
    x_hat = clip_to_T(x, Tclip);

    C_hat = sqrt(1 / L) * fft(x_hat, LN);
    c_in = C_hat(1:N);

    Tpeak = CR * norm(x_hat) / sqrt(LN);
    H = solve_projection(c, c_in, C_idx, A, L, N, Tpeak, useCVX, verbose, it);

    x = ofdm_synth(H .* c_in, L);
    x = clip_to_T(x, Tpeak);

    if verbose
        papr = max(abs(x).^2) / max(mean(abs(x).^2), eps);
        fprintf('  [Wang] iter %3d/%d: PAPR = %.4f dB\n', ...
            it, iters, 10 * log10(papr));
    end
end

C_full = sqrt(1 / L) * fft(x, LN);
C_opt = C_full(1:N);

end

function validate_indices(C_idx, R_idx, N)
if any(C_idx < 1) || any(C_idx > N) || any(C_idx ~= round(C_idx))
    error('Wang:BadCommIndex', ...
        'C_idx must contain integer indices in [1, N].');
end
if any(R_idx < 1) || any(R_idx > N) || any(R_idx ~= round(R_idx))
    error('Wang:BadRadarIndex', ...
        'R_idx must contain integer indices in [1, N].');
end
if ~isempty(intersect(C_idx, R_idx))
    error('Wang:OverlappingIndices', ...
        'C_idx and R_idx must not overlap.');
end
end

function A = build_ofdm_synthesis_matrix(N, L)
LN = L * N;
n = (0:LN-1).';
k = 0:(N-1);
A = (sqrt(L) / LN) * exp(1j * 2 * pi * (n * k) / LN);
end

function x = ofdm_synth(C, L)
N = numel(C);
LN = L * N;
x = ifft([C(:); zeros(LN - N, 1)], LN) * sqrt(L);
end

function H = solve_projection(c, c_in, C_idx, A, L, N, Tpeak, useCVX, verbose, it)
ok = false;
H = ones(N, 1);

% Wang/Luo is a full-codeword PAPR optimizer. Keep C_idx in the signature
% only to preserve the wrapper interface used by the ISAC simulations.
%#ok<INUSD>
evm_idx = (1:N).';

if useCVX && exist('cvx_begin', 'file') == 2
    try
        cvx_begin quiet
            variable H_var(N) complex
            variable t
            minimize(t)
            subject to
                norm(c(evm_idx) - H_var(evm_idx) .* c_in(evm_idx), 2) <= ...
                    norm(c(evm_idx), 2) * t
                abs(A * (H_var .* c_in)) <= Tpeak
        cvx_end

        ok = strcmpi(cvx_status, 'Solved') || ...
            strcmpi(cvx_status, 'Inaccurate/Solved');
        if ok
            H = H_var;
        elseif verbose
            fprintf('  [Wang] CVX status at iter %d: %s\n', it, cvx_status);
        end
    catch ME
        if verbose
            fprintf('  [Wang] CVX failed at iter %d: %s\n', it, ME.message);
        end
    end
end

if ~ok
    H = feasible_projection(c_in, L, N, Tpeak);
end
end

function H = feasible_projection(c_in, L, N, Tpeak)
LN = L * N;
spec = zeros(LN, 1);
spec(1:N) = c_in;
xt = ifft(spec, LN) * sqrt(L);
xt = clip_to_T(xt, Tpeak);
Cxt = sqrt(1 / L) * fft(xt, LN);
Cxt(N+1:end) = 0;
H = Cxt(1:N) ./ (c_in + 1e-12);
end

function y = clip_to_T(x, T)
y = x;
mag = abs(y);
idx = mag > T;
if any(idx)
    y(idx) = T * y(idx) ./ mag(idx);
end
end
