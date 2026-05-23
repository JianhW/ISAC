%% PISL simulation main entry
N = 128;
Nc = 64;
Nr = N - Nc;
modSC = 'QPSK';
monte = 1;           % Monte Carlo trials
e = 0.2;             % Communication energy ratio
rng(32);

% Proposed joint PISL/PAPR OCDM optimizer
mine_lambda = 0.0016;
mine_refine_lambda = 0.0004;
mine_l_norm_order = 100;
mine_iteration = 1000;
mine_refine_iteration = 300;
mine_min_iteration = 200;
mine_eps_stop = 1e-7;
mine_os_factor = 4;

% Mine update mode switch:
%   'fast'   : recommended default for large simulations. It uses only the
%              radar-sphere Riemannian/tangent-gradient update with monotone
%              line search. Fastest, usually enough for PISL simulation.
%   'hybrid' : accurate/strong mode. It also evaluates the MM candidate and
%              direct gradient candidate every iteration, then chooses the
%              best true objective descent. Slower, often better for final
%              PAPR.
%   'mm'     : pure MM sphere-projection update, closest to the derivation,
%              but can be slow/conservative for l=100.
%   'gradient': direct gradient + Riemannian gradient candidates.
%
% Quick switch examples:
%   Fast mode    : mine_update_mode = 'fast';   mine_mm_period = 0;
%   Accurate mode: mine_update_mode = 'hybrid'; mine_mm_period = 0;
%   Compromise   : mine_update_mode = 'fast';   mine_mm_period = 20;
%                  (adds the expensive MM candidate every 20 iterations)
mine_update_mode = 'fast';
mine_mm_period = 0;

waveforms = {
    'Proposed Algorithm';
    'Original OCDM';
    'Original OFDM';
    'OFDM Varshney';
    'OFDM New-ICF';
};

nWave = numel(waveforms);
Progress = waitbar(0, 'Progress...');

% Storage
pisl_all = cell(1, nWave);
time_signal_final = cell(1, nWave);
for w = 1:nWave
    pisl_all{w} = zeros(1, monte);
end

% Helper transforms
PISL_fun = @(x) calc_PISL(x);
FH = @(x) ifft(x(:)) * sqrt(N);
PsiH = DFnT(N)';

%% Monte Carlo loop
for m = 1:monte

    waitbar(m / monte, Progress, ...
        sprintf('Progress: %d/%d (%.1f%%)', m, monte, 100 * m / monte));

    % 1. Random subcarrier allocation
    rand_idx = randperm(N);
    idx_comm = sort(rand_idx(1:Nc)).';
    mask_comm = false(N, 1);
    mask_comm(idx_comm) = true;
    idx_radar = find(~mask_comm);

    % 2. Communication symbols
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

    % 3. Radar symbols
    Er = 1 - e;
    if Er <= 0
        error('Parameter e is too large and leaves no radar energy.');
    end

    % Use equal-amplitude radar symbols with random phase so all phase-only
    % optimizers are compared with the same input spectrum.
    zr = sqrt(Er / Nr) * exp(1j * 2 * pi * rand(Nr, 1));

    % 4. Composite frequency-domain vector
    z = zeros(N, 1);
    z(idx_comm) = zc;
    z(idx_radar) = zr;

    % 5. Generate each waveform
    for w = 1:nWave
        switch w
            case 1  % Mine
                z_mine = Mine(z, idx_comm, idx_radar, ...
                    'lambda', mine_lambda, ...
                    'adaptiveLambda', false, ...
                    'osFactor', mine_os_factor, ...
                    'maxIt', mine_iteration, ...
                    'minIt', mine_min_iteration, ...
                    'lNormOrder', mine_l_norm_order, ...
                    'epsStop', mine_eps_stop, ...
                    'lnormBound', 'current', ...
                    'etaMode', 'upper', ...
                    'UpdateMode', mine_update_mode, ...
                    'mmPeriod', mine_mm_period, ...
                    'maxBacktracking', 20, ...
                    'backtrackingShrink', 0.5, ...
                    'verbose', false);
                z_mine = Mine(z_mine, idx_comm, idx_radar, ...
                    'lambda', mine_refine_lambda, ...
                    'adaptiveLambda', false, ...
                    'osFactor', mine_os_factor, ...
                    'maxIt', mine_refine_iteration, ...
                    'minIt', 50, ...
                    'lNormOrder', mine_l_norm_order, ...
                    'epsStop', mine_eps_stop, ...
                    'lnormBound', 'current', ...
                    'etaMode', 'upper', ...
                    'UpdateMode', mine_update_mode, ...
                    'mmPeriod', mine_mm_period, ...
                    'maxBacktracking', 20, ...
                    'backtrackingShrink', 0.5, ...
                    'verbose', false);
                time_signal = PsiH * z_mine;

            case 2  % OCDM
                time_signal = PsiH * z;

            case 3  % OFDM
                % Standard OFDM baseline: use equal-amplitude active tones.
                % This makes the autocorrelation sidelobes drop to the
                % numerical floor after dB clipping.
                z_ofdm = exp(1j * angle(z)) / sqrt(N);
                time_signal = FH(z_ofdm);

            case 4  % Varshney
                % Use OutputMode='raw' to reproduce the legacy Varshney
                % sidelobe result that generated the lower sidelobe curve.
                z_varshney = Varshney(z, idx_comm, idx_radar, ...
                    'OutputMode', 'raw');
                time_signal = FH(z_varshney);

            case 5  % Wang
                z_wang = Wang(z, idx_comm, idx_radar, ...
                    'L', 4, ...
                    'Iter', 2, ...
                    'UseCVX', true);
                time_signal = FH(z_wang);
        end

        pisl_all{w}(m) = PISL_fun(time_signal);

        if m == monte
            time_signal_final{w} = time_signal;
        end
    end
end

if isgraphics(Progress)
    close(Progress);
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

%% Mean PISL output
pisl_mean = zeros(1, nWave);
for w = 1:nWave
    pisl_mean(w) = mean(pisl_all{w});
end

for w = 1:nWave
    fprintf('%s: Mean PISL = %.6e\n', waveforms{w}, pisl_mean(w));
end

%% Periodic autocorrelation comparison
lags = -(N - 1):(N - 1);
legend_handles = gobjects(1, nWave);

fig_ac = figure('Color', 'w', 'Position', [140, 140, 760, 560]);
ax_ac = axes(fig_ac);
hold(ax_ac, 'on');
box(ax_ac, 'on');
set(ax_ac, 'FontName', 'Times New Roman', ...
    'FontSize', 13, ...
    'LineWidth', 1.2, ...
    'TickDir', 'in', ...
    'TickLength', [0.018, 0.018]);

plot_floor_db = -120;

for w = 1:nWave
    x = time_signal_final{w};
    r = periodic_autocorr(x);
    r_bilateral = [conj(flipud(r(2:end))); r];
    r0 = r_bilateral(N);
    r_dB = 20 * log10(max(abs(r_bilateral) / max(abs(r0), 1e-12), 1e-12));
    r_dB(r_dB < plot_floor_db) = plot_floor_db;

    marker_idx = unique(round(linspace(1, numel(lags), ...
        min(n_markers, numel(lags)))));

    legend_handles(w) = plot(ax_ac, lags, r_dB, ...
        'LineStyle', line_styles{mod(w - 1, length(line_styles)) + 1}, ...
        'Color', plot_colors(mod(w - 1, size(plot_colors, 1)) + 1, :), ...
        'LineWidth', line_width, ...
        'Marker', marker_styles{mod(w - 1, length(marker_styles)) + 1}, ...
        'MarkerIndices', marker_idx, ...
        'MarkerSize', marker_size, ...
        'MarkerFaceColor', 'w');
end

grid(ax_ac, 'on');
ax_ac.GridAlpha = 0.18;
ax_ac.MinorGridAlpha = 0.10;
ax_ac.XMinorGrid = 'on';
ax_ac.YMinorGrid = 'on';
ax_ac.Layer = 'top';
xlim(ax_ac, [lags(1), lags(end)]);
ylim(ax_ac, [plot_floor_db, 0]);

xlabel(ax_ac, 'Time Lag', 'FontName', 'Times New Roman', ...
    'FontSize', 15, 'FontWeight', 'bold');
ylabel(ax_ac, 'Periodic Autocorrelation Level (dB)', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 15, 'FontWeight', 'bold');
title(ax_ac, sprintf('Periodic Autocorrelation (N = %d, N_c = %d, MC = %d)', ...
    N, Nc, monte), ...
    'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');

lgd = legend(ax_ac, legend_handles, waveforms, ...
    'Location', 'southwest', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 11);
lgd.Box = 'off';
hold(ax_ac, 'off');

fprintf('Done!\n');

function pisl = calc_PISL(x)
    x = x(:);
    r = periodic_autocorr(x);
    main_energy = abs(r(1))^2;
    if main_energy <= eps
        pisl = 0;
        return;
    end
    pisl = real(sum(abs(r).^2) / main_energy - 1);
    pisl = max(pisl, 0);
end

function r = periodic_autocorr(x)
    x = x(:);
    r = ifft(abs(fft(x)).^2);
end

function DFnT_matrix = DFnT(P)
    m = (0:P-1)' * ones(1, P);
    n = ones(P, 1) * (0:P-1);

    fresnel_kernel = exp(1j * pi / P * (m - n).^2);
    normalization = (1 / sqrt(P)) * exp(1j * pi / 4);
    DFnT_matrix = normalization * fresnel_kernel;
end
