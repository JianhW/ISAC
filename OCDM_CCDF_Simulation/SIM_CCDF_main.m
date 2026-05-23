%% OCDM / OFDM CCDF simulation
% Compare baseline and optimization methods under a unified CCDF setup.

clear; clc; close all;

script_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(script_dir);
if isempty(project_dir)
    project_dir = pwd;
end
addpath(fullfile(project_dir, 'Algorithms'));

%% Parameters
N = 128;                   % Total number of subcarriers
Nc = 24;                   % Number of communication subcarriers
Nr = N - Nc;               % Number of radar subcarriers
modSC = 'QPSK';            % Communication modulation
e = 0.2;                   % Communication energy ratio
monte = 10;                % Monte Carlo trials
l_norm_order = 100;        % l-norm order
iteration = 1000;          % Main iteration number
os_factor = 4;             % Oversampling factor for CCDF evaluation
base_random_seed = 1;      % Fixed per-trial seed base for reproducible parfor resume

%% Parallel Monte Carlo settings
use_parallel_monte_carlo = GetDefaultParallelMonteCarloSetting();
parallel_pool_type = 'processes';  % 'processes' or 'threads'
parallel_batch_size = 16;          % Number of Monte Carlo trials per parfor batch

%% Result saving / checkpoint resume interface
% Resume usage:
%   1. First run: keep resume_simulation = true. The script periodically saves papr_results and mc_done.
%   2. Continue later: keep key parameters unchanged, increase monte, and rerun. The script continues from mc_done + 1.
%   3. Start a new independent run: set resume_simulation = false, or change result_file_mode/manual_result_file.
resume_simulation = true;
save_every_trials = 50;
result_dir = script_dir;
figure_dir = fullfile(project_dir, 'figure');
result_file_mode = 'auto';  % 'auto': automatic file name; 'manual': use manual_result_file
manual_result_file = fullfile(result_dir, 'CCDF_my_experiment.mat');

% Parameters for the proposed OCDM algorithm in Mine.m
mine_lambda = 0.0016;
mine_warm_start_iteration = 500;
mine_pisl_guard = 1e-7;
mine_lnorm_bound = 'current';
mine_eig_mode = 'upper';
mine_max_backtracking = 14;
mine_lambda_shrink = 0.5;

% Mine update mode switch:
%   'fast'   : recommended default for CCDF/Monte Carlo runs. It uses only
%              the radar-sphere Riemannian/tangent-gradient update with
%              monotone line search, and skips the expensive MM candidate.
%   'hybrid' : accurate/strong mode. It evaluates the MM candidate, direct
%              gradient candidate, and Riemannian candidate every iteration,
%              then chooses the best true objective descent. Slower, often
%              better for final PAPR.
%   'mm'     : pure MM sphere-projection update, closest to the derivation,
%              but conservative for high l-norm orders such as l=100.
%   'gradient': direct gradient + Riemannian gradient candidates.
%
% Quick switch examples:
%   Fast mode    : mine_update_mode = 'fast';   mine_mm_period = 0;
%   Accurate mode: mine_update_mode = 'hybrid'; mine_mm_period = 0;
%   Compromise   : mine_update_mode = 'fast';   mine_mm_period = 20;
%                  (adds the expensive MM candidate every 20 iterations)
mine_update_mode = 'hybrid';
mine_mm_period = 0;

rng(base_random_seed);

if ~isfolder(result_dir)
    mkdir(result_dir);
end
if ~isfolder(figure_dir)
    mkdir(figure_dir);
end
if use_parallel_monte_carlo
    StartParallelPool(parallel_pool_type, project_dir);
end

%% Waveform list
waveforms = {
    'Mine',      'Proposed Algorithm';
    'OCDM',      'Original OCDM';
    'OFDM',      'Original OFDM';
    'Varshney',  'OFDM Varshney';
    'Wang',      'OFDM New-ICF';
};

nWave = size(waveforms, 1);
Progress = waitbar(0, 'Progress...');

waveform_file_tag = strjoin(waveforms(:, 1).', '_');
auto_result_name = sprintf('CCDF_ISAC_%s_N%d_Nc%d_e%.1f_l%d_Iter%d.mat', ...
    waveform_file_tag, N, Nc, e, l_norm_order, iteration);
switch lower(result_file_mode)
    case 'auto'
        result_file = fullfile(result_dir, auto_result_name);
    case 'manual'
        result_file = ResolveResultPath(manual_result_file, result_dir);
    otherwise
        error('result_file_mode must be ''auto'' or ''manual''.');
end
fprintf('CCDF checkpoint: %s\n', result_file);

%% Storage
papr_results = cell(1, nWave);
aisl_results = cell(1, nWave); 

for w = 1:nWave
    papr_results{w} = zeros(1, monte);
    aisl_results{w} = zeros(1, monte);
end
mc_done = 0;

%% =============杈呭姪鍑芥暟=============
PAPR_fun = @(x) max(abs(x(:)).^2) / mean(abs(x(:)).^2);
FH = @(x) ifft(x)*sqrt(N);
F = @(x) fft(x)*sqrt(N);

Psi_os = DFnT(os_factor * N);

if resume_simulation && isfile(result_file)
    saved_checkpoint = load(result_file);
    ValidateCheckpoint(saved_checkpoint, waveforms, N, Nc, e, modSC, ...
        l_norm_order, iteration, os_factor, mine_lambda, ...
        mine_pisl_guard, mine_lnorm_bound, mine_eig_mode, ...
        mine_max_backtracking, mine_lambda_shrink, mine_update_mode, ...
        mine_mm_period, base_random_seed);
    papr_results = saved_checkpoint.papr_results;
    mc_done = saved_checkpoint.mc_done;
    if mc_done > monte
        fprintf('Checkpoint has %d trials, current monte is %d. Truncating for plotting only.\n', ...
            mc_done, monte);
        mc_done = monte;
    end
    for w = 1:nWave
        if numel(papr_results{w}) < monte
            papr_results{w}(end + 1:monte) = 0;
        elseif numel(papr_results{w}) > monte
            papr_results{w} = papr_results{w}(1:monte);
        end
    end
    fprintf('Resuming CCDF simulation from %s, completed %d/%d Monte Carlo trials\n', ...
        result_file, mc_done, monte);
end

%% Monte Carlo loop
parallel_batch_size = max(1, min(parallel_batch_size, monte));
batch_start = mc_done + 1;
while batch_start <= monte
    batch_end = min(batch_start + parallel_batch_size - 1, monte);
    if batch_end > mc_done + save_every_trials
        batch_end = mc_done + save_every_trials;
    end
    batch_mc = batch_start:batch_end;
    nBatch = numel(batch_mc);
    papr_batch = zeros(nBatch, nWave);

    waitbar(batch_start / monte, Progress, ...
        sprintf('Progress: %d-%d/%d', batch_start, batch_end, monte));
    fprintf('Monte Carlo batch: %d-%d/%d\n', batch_start, batch_end, monte);

    if use_parallel_monte_carlo
        parfor ii = 1:nBatch
            mc = batch_mc(ii);
            papr_batch(ii, :) = RunOneMonteCarloTrial(mc, base_random_seed, ...
                N, Nc, Nr, modSC, e, nWave, os_factor, Psi_os, iteration, ...
                l_norm_order, mine_lambda, mine_pisl_guard, ...
                mine_lnorm_bound, mine_eig_mode, mine_max_backtracking, ...
                mine_lambda_shrink, mine_update_mode, mine_mm_period);
        end
    else
        for ii = 1:nBatch
            mc = batch_mc(ii);
            papr_batch(ii, :) = RunOneMonteCarloTrial(mc, base_random_seed, ...
                N, Nc, Nr, modSC, e, nWave, os_factor, Psi_os, iteration, ...
                l_norm_order, mine_lambda, mine_pisl_guard, ...
                mine_lnorm_bound, mine_eig_mode, mine_max_backtracking, ...
                mine_lambda_shrink, mine_update_mode, mine_mm_period);
        end
    end

    for w = 1:nWave
        papr_results{w}(batch_mc) = papr_batch(:, w).';
    end
    mc_done = batch_end;
    waitbar(mc_done / monte, Progress, ...
        sprintf('Progress: %d/%d (%.1f%%)', mc_done, monte, 100 * mc_done / monte));

    if mod(mc_done, save_every_trials) == 0 || mc_done == monte
        SaveCheckpoint(result_file, papr_results, mc_done, waveforms, ...
            N, Nc, Nr, modSC, e, monte, l_norm_order, iteration, os_factor, ...
            mine_lambda, mine_pisl_guard, mine_lnorm_bound, mine_eig_mode, ...
            mine_max_backtracking, mine_lambda_shrink, mine_update_mode, ...
            mine_mm_period, base_random_seed);
        fprintf('  Checkpoint saved: %d/%d Monte Carlo trials\n', mc_done, monte);
    end

    batch_start = batch_end + 1;
end

SaveCheckpoint(result_file, papr_results, mc_done, waveforms, ...
    N, Nc, Nr, modSC, e, monte, l_norm_order, iteration, os_factor, ...
    mine_lambda, mine_pisl_guard, mine_lnorm_bound, mine_eig_mode, ...
    mine_max_backtracking, mine_lambda_shrink, mine_update_mode, ...
    mine_mm_period, base_random_seed);

%% CCDF plotting
if isgraphics(Progress)
    close(Progress);
end

thresholds = 0:0.1:12;
ccdf_curves = cell(1, nWave);

for w = 1:nWave
    papr_dB = 10 * log10(papr_results{w});
    ccdf_curves{w} = arrayfun(@(t) mean(papr_dB > t), thresholds);
end

plot_colors = [
    0.7200, 0.2200, 0.1200;
    0.0000, 0.2700, 0.5200;
    0.0000, 0.5000, 0.3200;
    0.4500, 0.1800, 0.6200;
    0.7800, 0.5500, 0.0000
];
line_styles = {'-', '-', '-', '-', '-'};
marker_styles = {'h', 'o', 's', '^', 'd'};
line_width = 3.0;
marker_size = 10.5;
n_markers = 11;

legend_handles = gobjects(1, nWave);

fig = figure('Color', 'w', 'Position', [120, 120, 760, 560]);
ax = axes(fig);
hold(ax, 'on');
box(ax, 'on');
set(ax, 'FontName', 'Times New Roman', ...
        'FontSize', 13, ...
        'LineWidth', 1.2, ...
        'TickDir', 'in', ...
        'TickLength', [0.018, 0.018], ...
        'YScale', 'log');

for w = 1:nWave
    ccdf_plot = ccdf_curves{w};
    ccdf_plot(ccdf_plot == 0) = NaN;

    valid_idx = find(~isnan(ccdf_plot));
    
    if ~isempty(valid_idx)

        x_valid = thresholds(valid_idx);
    
        y_valid = log10(ccdf_plot(valid_idx));
   
        dx = diff(x_valid);
        dy = diff(y_valid);
    
        seg_len = sqrt(dx.^2 + dy.^2);
    
        cum_len = [0 cumsum(seg_len)];

        target_len = linspace(0, cum_len(end), ...
            min(n_markers, length(valid_idx)));

        marker_local_idx = arrayfun(@(s) ...
            find(abs(cum_len - s) == min(abs(cum_len - s)), 1), ...
            target_len);

        marker_idx = valid_idx(unique(marker_local_idx));
    
    else
        marker_idx = [];
    
    end

    legend_handles(w) = semilogy(ax, thresholds, ccdf_plot, ...
        'LineStyle', line_styles{mod(w - 1, length(line_styles)) + 1}, ...
        'Color', plot_colors(mod(w - 1, size(plot_colors, 1)) + 1, :), ...
        'LineWidth', line_width, ...
        'Marker', marker_styles{mod(w - 1, length(marker_styles)) + 1}, ...
        'MarkerIndices', marker_idx, ...
        'MarkerSize', marker_size, ...
        'MarkerFaceColor', 'w');
end

grid(ax, 'on');
ax.GridAlpha = 0.18;
ax.MinorGridAlpha = 0.10;
ax.XMinorGrid = 'on';
ax.YMinorGrid = 'on';
ax.Layer = 'top';
xlim(ax, [thresholds(1), thresholds(end)]);

valid_curves = ccdf_curves(cellfun(@(c) any(c > 0), ccdf_curves));
if ~isempty(valid_curves)
    min_positive_ccdf = min(cellfun(@(c) min(c(c > 0)), valid_curves));
    if min_positive_ccdf < 1
        y_lower = 10^(floor(log10(min_positive_ccdf)));
    else
        y_lower = 0.1;
    end
    ylim(ax, [y_lower, 1]);
end

xlabel(ax, 'PAPR (dB)', 'FontName', 'Times New Roman', 'FontSize', 15, 'FontWeight', 'bold');
ylabel(ax, 'CCDF', 'FontName', 'Times New Roman', 'FontSize', 15, 'FontWeight', 'bold');
title(ax, sprintf('CCDF of PAPR (N = %d, N_c = %d, l = %d, Iter = %d, MC = %d)', ...
    N, Nc, l_norm_order, iteration, monte), ...
    'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');

lgd = legend(ax, legend_handles, waveforms(:, 2), ...
    'Location', 'southwest', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 11);
lgd.Box = 'off';

target_ccdf = 1e-4;
for w = 1:nWave
    ccdf_current = ccdf_curves{w};
    [~, idx] = min(abs(ccdf_current - target_ccdf));
    if ~isempty(idx) && idx >= 1 && idx <= length(thresholds)
        fprintf('%s @ nearest CCDF to 1e-4: PAPR = %.2f dB, CCDF = %.4g\n', ...
            waveforms{w, 2}, thresholds(idx), ccdf_current(idx));
    end
end

print(fig, sprintf('CCDF_N%d_Nc%d_e%.1f_lambda%.3g.png', N, Nc, e, mine_lambda), '-dpng', '-r600');
savefig(fig, sprintf('CCDF_main_first1.fig'));
hold(ax, 'off');

fprintf('Done!\n');

function use_parallel = GetDefaultParallelMonteCarloSetting()
use_parallel = true;
end

function DFnT_matrix = DFnT(P)
m = (0:P-1)' * ones(1, P);
n = ones(P, 1) * (0:P-1);
fresnel_kernel = exp(1j * pi / P * (m - n).^2);
normalization = (1 / sqrt(P)) * exp((pi / 4) * 1j);
DFnT_matrix = normalization * fresnel_kernel;
end

function papr_values = RunOneMonteCarloTrial(mc, base_random_seed, N, Nc, Nr, ...
    modSC, e, nWave, os_factor, Psi_os, iteration, l_norm_order, ...
    mine_lambda, mine_pisl_guard, mine_lnorm_bound, mine_eig_mode, ...
    mine_max_backtracking, mine_lambda_shrink, mine_update_mode, mine_mm_period)
rng(GetTrialSeed(base_random_seed, mc), 'twister');
PAPR_fun = @(x) max(abs(x(:)).^2) / mean(abs(x(:)).^2);
FH = @(x) ifft(x)*sqrt(N);

rand_idx = randperm(N);
idx_comm = sort(rand_idx(1:Nc)).';
mask_comm = false(N, 1);
mask_comm(idx_comm) = true;
idx_radar = find(~mask_comm);

switch upper(modSC)
    case 'BPSK'
        bits = randi([0 1], Nc, 1);
        zc_unit = exp(1j * pi * bits);
    case 'QPSK'
        bits = randi([0 1], 2 * Nc, 1);
        zc_unit = (1 / sqrt(2)) * ...
            ((2 * bits(1:2:end) - 1) + 1j * (2 * bits(2:2:end) - 1));
    otherwise
        error('Only BPSK/QPSK are supported.');
end

zc = sqrt(e) * zc_unit / sqrt(Nc);
Er = 1 - e;
if Er <= 0
    error('Parameter e is too large and leaves no radar energy.');
end
zr = randn(Nr, 1) + 1j * randn(Nr, 1);
zr = zr / norm(zr) * sqrt(Er);

z = zeros(N, 1);
z(idx_comm) = zc;
z(idx_radar) = zr;

papr_values = zeros(1, nWave);
for w = 1:nWave
    switch w
        case 1
            z_mine = Mine(z, idx_comm, idx_radar, ...
                'maxIt', iteration, ...
                'minIt', 200, ...
                'lNormOrder', l_norm_order, ...
                'lambda', mine_lambda, ...
                'osFactor', os_factor, ...
                'epsStop', mine_pisl_guard, ...
                'lnormBound', mine_lnorm_bound, ...
                'etaMode', mine_eig_mode, ...
                'maxBacktracking', mine_max_backtracking, ...
                'backtrackingShrink', mine_lambda_shrink, ...
                'UpdateMode', mine_update_mode, ...
                'mmPeriod', mine_mm_period, ...
                'PaprWeight', 0.85);
            z_padded = zeros(os_factor * N, 1);
            z_padded(1:N) = z_mine;
            time_signal = Psi_os' * z_padded * sqrt(os_factor * N);
        case 2
            z_padded = zeros(os_factor * N, 1);
            z_padded(1:N) = z;
            time_signal = Psi_os' * z_padded * sqrt(os_factor * N);
        case 3
            z_padded = [z; zeros((os_factor - 1) * N, 1)];
            time_signal = ifft(z_padded) * sqrt(os_factor * N);
        case 4
            time_signal = Varshney(z, idx_comm, idx_radar);
        case 5
            z_wang = Wang(z, idx_comm, idx_radar, ...
                'L', 4, ...
                'Iter', 2, ...
                'UseCVX', true);
            time_signal = FH(z_wang);
    end
    papr_values(w) = PAPR_fun(time_signal);
end
end

function seed = GetTrialSeed(base_random_seed, mc)
seed = mod(double(base_random_seed) + 1000003 * double(mc), 2^32 - 1);
if seed == 0
    seed = 1;
end
end

function StartParallelPool(pool_type, project_dir)
pool = gcp('nocreate');
if ~isempty(pool)
    return;
end
switch lower(pool_type)
    case 'processes'
        pool = parpool('Processes');
    case 'threads'
        pool = parpool('Threads');
    otherwise
        error('parallel_pool_type must be ''processes'' or ''threads''.');
end
if isa(pool, 'parallel.ProcessPool')
    addAttachedFiles(pool, {
        fullfile(project_dir, 'Algorithms', 'Mine.m'), ...
        fullfile(project_dir, 'Algorithms', 'Varshney.m'), ...
        fullfile(project_dir, 'Algorithms', 'Wang.m')});
end
end

function SaveCheckpoint(result_file, papr_results, mc_done, waveforms, ...
    N, Nc, Nr, modSC, e, monte, l_norm_order, iteration, ...
    os_factor, mine_lambda, mine_pisl_guard, mine_lnorm_bound, ...
    mine_eig_mode, mine_max_backtracking, mine_lambda_shrink, ...
    mine_update_mode, mine_mm_period, base_random_seed)
save(result_file, ...
    'papr_results', 'mc_done', 'waveforms', ...
    'N', 'Nc', 'Nr', 'modSC', 'e', 'monte', 'l_norm_order', ...
    'iteration', 'os_factor', 'mine_lambda', 'mine_pisl_guard', ...
    'mine_lnorm_bound', 'mine_eig_mode', 'mine_max_backtracking', ...
    'mine_lambda_shrink', 'mine_update_mode', 'mine_mm_period', ...
    'base_random_seed');
end

function ValidateCheckpoint(saved, waveforms, N, Nc, e, modSC, ...
    l_norm_order, iteration, os_factor, mine_lambda, mine_pisl_guard, ...
    mine_lnorm_bound, mine_eig_mode, mine_max_backtracking, ...
    mine_lambda_shrink, mine_update_mode, mine_mm_period, base_random_seed)
required_fields = {'papr_results', 'mc_done', 'waveforms', 'N', 'Nc', ...
    'e', 'modSC', 'l_norm_order', 'iteration', 'os_factor', ...
    'mine_lambda', 'mine_pisl_guard', 'mine_lnorm_bound', ...
    'mine_eig_mode', 'mine_max_backtracking', 'mine_lambda_shrink', ...
    'mine_update_mode', 'mine_mm_period', 'base_random_seed'};
for k = 1:numel(required_fields)
    if ~isfield(saved, required_fields{k})
        error('Checkpoint file does not contain %s.', required_fields{k});
    end
end
if ~isequal(saved.waveforms, waveforms)
    error('Checkpoint waveforms do not match the current simulation.');
end
if saved.N ~= N
    error('Checkpoint N does not match the current simulation.');
end
if saved.Nc ~= Nc
    error('Checkpoint Nc does not match the current simulation.');
end
if abs(saved.e - e) > eps
    error('Checkpoint communication energy ratio e does not match the current simulation.');
end
if ~strcmpi(saved.modSC, modSC)
    error('Checkpoint modulation does not match the current simulation.');
end
if saved.l_norm_order ~= l_norm_order
    error('Checkpoint l_norm_order does not match the current simulation.');
end
if saved.iteration ~= iteration
    error('Checkpoint iteration does not match the current simulation.');
end
if saved.os_factor ~= os_factor
    error('Checkpoint os_factor does not match the current simulation.');
end
if abs(saved.mine_lambda - mine_lambda) > eps
    error('Checkpoint mine_lambda does not match the current simulation.');
end
if abs(saved.mine_pisl_guard - mine_pisl_guard) > eps
    error('Checkpoint mine_pisl_guard does not match the current simulation.');
end
if ~strcmp(saved.mine_lnorm_bound, mine_lnorm_bound)
    error('Checkpoint mine_lnorm_bound does not match the current simulation.');
end
if ~strcmp(saved.mine_eig_mode, mine_eig_mode)
    error('Checkpoint mine_eig_mode does not match the current simulation.');
end
if saved.mine_max_backtracking ~= mine_max_backtracking
    error('Checkpoint mine_max_backtracking does not match the current simulation.');
end
if abs(saved.mine_lambda_shrink - mine_lambda_shrink) > eps
    error('Checkpoint mine_lambda_shrink does not match the current simulation.');
end
if ~strcmp(saved.mine_update_mode, mine_update_mode)
    error('Checkpoint mine_update_mode does not match the current simulation.');
end
if saved.mine_mm_period ~= mine_mm_period
    error('Checkpoint mine_mm_period does not match the current simulation.');
end
if saved.base_random_seed ~= base_random_seed
    error('Checkpoint base_random_seed does not match the current simulation.');
end
end

function result_file = ResolveResultPath(result_file, result_dir)
if isempty(result_file)
    error('Result file path cannot be empty.');
end
if ~isfolder(result_dir)
    mkdir(result_dir);
end
[folder, ~, ~] = fileparts(result_file);
if isempty(folder)
    result_file = fullfile(result_dir, result_file);
end
end
