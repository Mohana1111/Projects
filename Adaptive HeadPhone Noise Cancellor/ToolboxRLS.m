classdef ToolboxRLS < handle
    properties
        forgettingFactor;
        filterSize;
        filterWeights;
        weightHistory;
        tonalEnable;
        tonalFreqs;
        tonalBandWidths;
        tonalFilterSize;
        cleanSpeech;

        x;
        P;
        E;

        xn_1; xn_2; yn_1; yn_2

        delta = 1e-4;
        beta ;
        x_nlms;
        E_nlms;
        nlms_filterWeights;

        runIdealCase;
        historyMSE;
        historySNR;
        idealWeights;
        P_ideal;
        SNR_gain;
        plotSNRHistory;

        plotTime;
        plotFreq;
        plotTimeFreq;
        plotEvolution;

        fs;
        sampleCount;
    end
    methods(Static)
        function val = setValue(structInput, fieldName, defaultVal)
            if isfield(structInput, fieldName)
                val = structInput.(fieldName);
            else
                val = defaultVal;
            end
        end

        function [b, a] = designNotch(w0, bw, filterOrder)

            % w0: normalized notch frequency (0 < w0 < 1)
            % bw: normalized bandwidth (0 < bw < 1)
            % filterOrder: filter order (must be even)

            if mod(filterOrder, 2) ~= 0
                filterOrder = filterOrder + 1; % make it even
                warning('Filter order must be even. Using order %d instead.', filterOrder);
            end
            
            r = 1-(bw/2); 
            halfOrder = filterOrder/2;
            b = 1;
            a = 1;
            
            for k = 1:halfOrder
                b_section = [1, -2*cos(pi*w0), 1]; 
                a_section = [1, -2*r*cos(pi*w0), r^2]; 
                
                b = conv(b, b_section);
                a = conv(a, a_section);
            end
        end
    end
 
    methods
        function obj = ToolboxRLS(config)
            if nargin == 0 || ~isstruct(config)
                error('Constructor requires a struct as input.');
            end

            obj.forgettingFactor = ToolboxRLS.setValue(config, 'forgettingFactor', 0.99);
            obj.filterSize = ToolboxRLS.setValue(config, 'filterSize', 32);
            obj.filterWeights = ToolboxRLS.setValue(config, 'filterWeights', zeros(config.filterSize, 1));
            obj.weightHistory = [];
            obj.x = zeros(obj.filterSize, 1);
            obj.P = (1 / obj.delta) * eye(obj.filterSize);


            obj.tonalEnable = ToolboxRLS.setValue(config, 'tonalEnable', false);
            obj.tonalFreqs = ToolboxRLS.setValue(config, 'tonalFreqs', []);
            obj.tonalFilterSize = ToolboxRLS.setValue(config, 'tonalFilterSize', 16);
            obj.cleanSpeech = ToolboxRLS.setValue(config, 'cleanSpeech', []);

            obj.tonalBandWidths = ToolboxRLS.setValue(config, 'tonalBandWidths', 20*ones(1, length(obj.tonalFreqs)));

            obj.delta = ToolboxRLS.setValue(config, 'delta', 1e-4);
            obj.runIdealCase = ToolboxRLS.setValue(config, 'runIdealCase', false);
            obj.historyMSE = ToolboxRLS.setValue(config, 'historyMSE', []);
            obj.idealWeights = zeros(obj.filterSize, 1);

            % Updated plotting flags with more explicit names
            obj.plotTime = ToolboxRLS.setValue(config, 'plotTime', true);
            obj.plotFreq = ToolboxRLS.setValue(config, 'plotFreq', true);
            obj.plotTimeFreq = ToolboxRLS.setValue(config, 'plotTimeFreq', true);
            obj.plotEvolution = ToolboxRLS.setValue(config, 'plotFilterEvolution', true);            obj.plotSNRHistory = ToolboxRLS.setValue(config, 'plotSNRHistory', true);
            
            obj.fs = ToolboxRLS.setValue(config, 'fs', 44100);

            obj.E = [];
            obj.x_nlms = [];

            obj.sampleCount = 1;
        end



        function y = recursiveUpdate(obj, x_sample, d_sample)
            
            % Update the buffer
            obj.x = [x_sample; obj.x(1:end-1)];  

            % if obj.tonalEnable
            %     obj.x = obj.applyNotchFilter(obj.x, obj.fs);
            % end


            % Calculate Gain Vector
            u = obj.P * obj.x; 
            k = u / (obj.forgettingFactor + obj.x' * u);  

            % Calculate Output based on previous filter weights
            y = obj.filterWeights' * obj.x; 
            
            % Calculate Priori Error
            e = d_sample - y;
            obj.E(obj.sampleCount) = e;

            % Update Weights
            obj.filterWeights = obj.filterWeights + k * e; 

            % Update Inverse Covariance Matrix
            obj.P = (obj.P - k * (obj.x') * obj.P) / obj.forgettingFactor; 

            obj.sampleCount = obj.sampleCount + 1;

            obj.weightHistory(:, obj.sampleCount) = obj.filterWeights;
            obj.historyMSE(obj.sampleCount) = e^2;
        end
        
        function nlmsUpdate(obj , x_sample, d_sample)
            f_order = obj.filterSize;
            norm_x = 0 ;
            
            x_buf = zeros(f_order, 1);

            k = obj.sampleCount ; 
            

            if k <= f_order
                x_new = x_sample;
                x_old = 0;
                
                
                x_buf = [x_new; x_buf(1:end-1)];
            else
                x_new = x_sample;
                x_old = x_buf(k - f_order);
                
                
                x_buf = [x_new; x_buf(1:end-1)];
            end
            
            norm_x = norm_x + abs(x_new)^2 - abs(x_old)^2;
        
        
            e = d_sample - obj.nlms_filterWeights' * obj.x;
            obj.E_nlms(obj.sampleCount) = e ;
            
            
            obj.nlms_filterWeights =obj.nlms_filterWeights + (obj.beta / (norm_x + 1e-4)) * e * conj(x_buf).';
            
        end

        function calculateIdealWeights(obj, d_sample)
            % we want to see the performance of the filter, when we know the clean speech.

            if isempty(obj.x) || isempty(obj.P_ideal)
                obj.x = zeros(obj.filterSize, 1);
                obj.P_ideal = (1 / obj.delta) * eye(obj.filterSize);
            end

            d_sample = d_sample - obj.cleanSpeech(obj.sampleCount);
            % obj.x = [x_sample; obj.x(1:end-1)];  

            u = obj.P_ideal * obj.x; 
            k = u / (obj.forgettingFactor + obj.x' * u);  

            y = obj.idealWeights' * obj.x; 
            
            e = d_sample - y;

            obj.idealWeights = obj.idealWeights + k * e; 

            obj.P_ideal = (obj.P_ideal - k * (obj.x') * obj.P_ideal) / obj.forgettingFactor; 
        end

        function outputSignal = applyNotchFilter(obj, inputSignal)
            if mod(obj.tonalFilterSize, 2) ~= 0
                obj.tonalFilterSize = obj.tonalFilterSize + 1;
                warning('Filter order must be even. Using order %d instead.', obj.tonalFilterSize);
            end

             outputSignal = inputSignal;

            for i = 1:length(obj.tonalFreqs)
                w0 = obj.tonalFreqs(i) / (obj.fs / 2);
                bw = obj.tonalBandWidths(i) / (obj.fs / 2);

                [b_notch, a_notch] = ToolboxRLS.designNotch(w0, bw, obj.tonalFilterSize);

                outputSignal = filter(b_notch, a_notch, inputSignal);
                
            end
        end

        function y = notchRecursive(obj, x)
            % Initialize delay buffers if empty
            if isempty(obj.xn_1)
                numFilters = length(obj.tonalFreqs);
                obj.xn_1 = zeros(numFilters, 1);
                obj.xn_2 = zeros(numFilters, 1);
                obj.yn_1 = zeros(numFilters, 1);
                obj.yn_2 = zeros(numFilters, 1);
            end
        
            y = x;
        
            for i = 1:length(obj.tonalFreqs)
                w0 = obj.tonalFreqs(i) / (obj.fs / 2);  % Normalized frequency
                bw = obj.tonalBandWidths(i) / (obj.fs / 2);
        
                [b, a] = ToolboxRLS.designNotch(w0, bw, 2);  % Design 2nd-order notch
        
                % Apply difference equation
                x_n = y;  % Apply filtering in sequence to accumulate effects
                y = b(1)*x_n + ...
                    b(2)*obj.xn_1(i) + ...
                    b(3)*obj.xn_2(i) - ...
                    a(2)*obj.yn_1(i) - ...
                    a(3)*obj.yn_2(i);
        
                % Update delay buffers for this filter
                obj.xn_2(i) = obj.xn_1(i);
                obj.xn_1(i) = x_n;
        
                obj.yn_2(i) = obj.yn_1(i);
                obj.yn_1(i) = y;
            end
        end


        function resetFilter(obj)
            obj.filterWeights = zeros(obj.filterSize, 1);
            obj.weightHistory = [];
            obj.historyMSE = [];
            obj.sampleCount = 0;
            obj.x = [];
            obj.P = [];
            obj.E = [];
            obj.idealWeights = zeros(obj.filterSize, 1);
            obj.P_ideal = [];
        end

        function updateSNRHistory(obj)
            if isempty(obj.historySNR)
                obj.historySNR = zeros(1, 1e5);
            end
        
            idx = obj.sampleCount - 1;
        
            if idx <= length(obj.cleanSpeech)

                cleanSpeechChunk = obj.cleanSpeech(1:idx);
                errorHistoryChunk = obj.E(1:idx);
                
                noisePower = sum((errorHistoryChunk' - cleanSpeechChunk).^2);
                signalPower = sum(cleanSpeechChunk.^2);

                % Some debug prints to check 
                % disp(size(obj.cleanSpeech(1:idx)));
                % disp(size(obj.historyMSE(1:idx)));
                % disp(size(signalPower));
                % disp(size(noisePower));
        
                if noisePower == 0
                    snr = Inf;  % check for perfect cancellation - never gonna happen tho
                else
                    snr = 10 * log10(signalPower / noisePower);
                end
        
                obj.historySNR(idx) = snr;
            else
                obj.historySNR(idx) = NaN;
            end
        end

        function computeRNTN(obj, d)
            if ~obj.tonalEnable
                disp("Tonal Retention not Enabled, Skipping Residual Non Tonal Noise Suppression");
                return;
            end
            N = length(d);
            f = (0:floor(N/2)) * (obj.fs / N);

            D_FFT = abs(fft(d));
            D_FFT = D_FFT(1:length(f));
        
            Clean_FFT = abs(fft(obj.cleanSpeech));
            Clean_FFT = Clean_FFT(1:length(f));
        
            E_FFT = abs(fft(obj.E));
            E_FFT = E_FFT(1:length(f));
    
            tonal_noise = zeros(length(obj.tonalFreqs), 1);
            count = 1;

            for i = 1:length(obj.tonalFreqs)
                [~, idx] = min(abs(f - obj.tonalFreqs(i)));
                disp(obj.tonalFreqs(i));
                disp(D_FFT(idx));
                tonal_noise(count) = D_FFT(idx);
                count= count + 1;
            end

            D_FFT = fft(d);

            tonal_power = sum(abs(tonal_noise).^2);
            total_power = sum(abs(D_FFT).^2);

            clean_power = sum(abs(Clean_FFT).^2);
            non_tonal_power = total_power - clean_power - tonal_power;

            output_signal_power = sum(abs(E_FFT).^2);

            % disp(size(output_signal_power))
            disp( output_signal_power);
            disp( clean_power);
            disp( tonal_power);
            disp( non_tonal_power);
            disp( total_power);

            % disp(output_signal_power - clean_power - tonal_power);

            residualNonTonalNoisePower = abs(output_signal_power - clean_power - tonal_power) / non_tonal_power;

            disp("Normalised Residual Non-Tonal Noise Power: " + residualNonTonalNoisePower);
        end
        
        function computeSNR(obj , d)
            % SNR before Filtering
            noise_before = d - obj.cleanSpeech;  
            P_signal = sum(obj.cleanSpeech .^ 2);
            P_noise_before = sum(noise_before .^ 2);
            % disp(size(P_noise_before));
            % disp(size(P_signal));
            SNR_pre = 10 * log10(P_signal / P_noise_before);
            fprintf('SNR Before Filtering = %.2f dB\n', SNR_pre);

            %  SNR After Filtering
            residual_noise = obj.E' - obj.cleanSpeech;
            P_noise_after = sum(residual_noise .^ 2);
            SNR_post = 10 * log10(P_signal / P_noise_after);
            fprintf('SNR After Filtering = %.2f dB\n', SNR_post);

            obj.SNR_gain = SNR_post - SNR_pre;
            fprintf('SNR Gain = %.2f dB\n', obj.SNR_gain);
        end

        function plotTimeDomain(obj, inputSignal, desiredSignal)
            figure('Name', 'Time-Domain Signals', 'Color', 'white', 'Position', [100, 100, 1000, 800]);

                % Time vector
                t = (0:length(inputSignal)-1) / obj.fs;

                % External Noise
                subplot(4,1,1);
                plot(t, inputSignal, 'LineWidth', 2, 'Color', [0.2 0.2 0.2]);
                title('External Noise', 'FontSize', 12, 'FontWeight', 'bold');
                ylabel('Amplitude', 'FontWeight', 'bold');
                grid on;
                set(gca, 'FontSize', 10);

                % Noisy Speech
                subplot(4,1,2);
                plot(t, desiredSignal, 'LineWidth', 2, 'Color', [0 0.5 0]);
                title('Noisy Speech', 'FontSize', 12, 'FontWeight', 'bold');
                ylabel('Amplitude', 'FontWeight', 'bold');
                grid on;
                set(gca, 'FontSize', 10);

                % Clean Speech
                subplot(4,1,3);
                plot(t, obj.cleanSpeech, 'LineWidth', 2, 'Color', [0 0 0.8]);
                title('Clean Speech', 'FontSize', 12, 'FontWeight', 'bold');
                ylabel('Amplitude', 'FontWeight', 'bold');
                grid on;
                set(gca, 'FontSize', 10);

                % Enhanced Speech
                subplot(4,1,4);
                plot(t, obj.E, 'LineWidth', 2, 'Color', [0.8 0 0]);
                title('Enhanced Speech', 'FontSize', 12, 'FontWeight', 'bold');
                xlabel('Time (s)', 'FontWeight', 'bold');
                ylabel('Amplitude', 'FontWeight', 'bold');
                grid on;
                set(gca, 'FontSize', 10);
        end
        
        function plotFrequencyDomain(obj, externalNoise, desiredSignal)
            if ~obj.plotFreq
                return;
            end
        
            figure('Name', 'Frequency Domain Comparison', 'Color', 'white', 'Position', [100, 100, 1000, 1200]);

                N = length(externalNoise);
                f = linspace(0, obj.fs/2, floor(N/2)+1);

                % FFT and magnitude
                Y_noise = fft(externalNoise);
                Y_desired = fft(desiredSignal);
                Y_clean = fft(obj.cleanSpeech);
                Y_enhanced = fft(obj.E);

                % Linear magnitude
                mag_noise = abs(Y_noise(1:floor(N/2)+1));
                mag_desired = abs(Y_desired(1:floor(N/2)+1));
                mag_clean = abs(Y_clean(1:floor(N/2)+1));
                mag_enhanced = abs(Y_enhanced(1:floor(N/2)+1));

                subplot(4,1,1);
                plot(f, mag_noise, 'LineWidth', 2, 'Color', [0.2 0.2 0.2]);
                xlabel('Frequency (Hz)', 'FontWeight', 'bold');
                ylabel('Magnitude', 'FontWeight', 'bold');
                title('External Noise Spectrum', 'FontWeight', 'bold');
                grid on;
                set(gca, 'FontSize', 10);

                subplot(4,1,2);
                plot(f, mag_desired, 'LineWidth', 2, 'Color', [0 0.5 0]);
                xlabel('Frequency (Hz)', 'FontWeight', 'bold');
                ylabel('Magnitude', 'FontWeight', 'bold');
                title('Noisy Speech Spectrum', 'FontWeight', 'bold');
                grid on;
                set(gca, 'FontSize', 10);

                subplot(4,1,3);
                plot(f, mag_clean, 'LineWidth', 2, 'Color', [0 0 0.8]);
                xlabel('Frequency (Hz)', 'FontWeight', 'bold');
                ylabel('Magnitude', 'FontWeight', 'bold');
                title('Clean Speech Spectrum', 'FontWeight', 'bold');
                grid on;
                set(gca, 'FontSize', 10);

                subplot(4,1,4);
                plot(f, mag_enhanced, 'LineWidth', 2, 'Color', [0.8 0 0]);
                xlabel('Frequency (Hz)', 'FontWeight', 'bold');
                ylabel('Magnitude', 'FontWeight', 'bold');
                title('Enhanced Speech Spectrum', 'FontWeight', 'bold');
                grid on;
                set(gca, 'FontSize', 10);

        end

        function plotFilterEvolution(obj)
            if ~obj.plotEvolution || isempty(obj.weightHistory)
                return;
            end
        
            figure('Name', 'Filter Coefficients and Frequency Response Evolution', ...
                   'Color', 'white', 'Position', [100, 100, 1600, 800]);
        
            % Time vector for plotting
            t = (0:size(obj.weightHistory, 2)-1) / obj.fs;
        
            % === Subplot 1: Filter Coefficient Evolution ===
            subplot(2, 2, [1 3]);
            plot(t, obj.weightHistory', 'LineWidth', 1);
            title('Evolution of All Filter Coefficients', 'FontWeight', 'bold');
            xlabel('Time (s)', 'FontWeight', 'bold');
            ylabel('Coefficient Value', 'FontWeight', 'bold');
            grid on;
        
            % === Subplot 2: Overlaid 2D Frequency Responses ===
            subplot(2, 2, [2, 4]);
            hold on;
        
            % Parameters
            nfft = 1024;
            f = linspace(0, obj.fs/2, nfft/2);
            step = 1000;
            numSnapshots = floor(size(obj.weightHistory, 2) / step);
            cmap = jet(numSnapshots);
        
            % Plot overlaid responses
            colorIdx = 1;
            for i = 1:step:size(obj.weightHistory, 2)
                h = freqz(obj.weightHistory(:, i), 1, nfft, obj.fs);
                plot(f, 20*log10(abs(h(1:nfft/2))), 'Color', cmap(colorIdx, :), 'LineWidth', 1);
                colorIdx = colorIdx + 1;
                if colorIdx > size(cmap, 1)
                    break;  % Avoid index overflow
                end
            end
        
            title('Frequency Response Evolution (Overlaid)', 'FontWeight', 'bold');
            xlabel('Frequency (Hz)', 'FontWeight', 'bold');
            ylabel('Magnitude (dB)', 'FontWeight', 'bold');
            grid on;
        
            % Colorbar with time ticks
            colormap(gca, cmap);
            cb = colorbar('Ticks', linspace(0, 1, 5), ...
                          'TickLabels', round(linspace(0, t(end), 5), 2));
            cb.Label.String = 'Time (s)';
            cb.Label.FontWeight = 'bold';
        

        
            % Overall plot styling
            set(gcf, 'Color', 'white');
            set(findall(gcf, '-property', 'FontSize'), 'FontSize', 10);
        end
    
        function plotSNR(obj)
            if ~obj.plotSNRHistory || isempty(obj.historySNR)
                return;
            end
    
            figure('Name', 'SNR History', 'Color', 'white', 'Position', [100, 100, 1000, 600]);
    
            validIdx = find(~isnan(obj.historySNR));
            t = (validIdx) / obj.fs;
            snrValues = obj.historySNR(validIdx);
    
            plot(t, snrValues, 'LineWidth', 2, 'Color', [0.2 0.4 0.6]);
            xlabel('Time (s)', 'FontWeight', 'bold');
            ylabel('SNR (dB)', 'FontWeight', 'bold');
            title('Signal-to-Noise Ratio Over Time', 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            set(gca, 'FontSize', 10);
        end
        
        function plotAllResults(obj, inputSignal, desiredSignal)
            % Consolidated plotting function to generate all available plots
            % Check if any plotting is enabled
            if ~(obj.plotTime || obj.plotFreq || obj.plotTimeFreq || ...
                 obj.plotEvolution || obj.plotSNRHistory)
                warning('No plotting options are enabled. Use config to enable plots.');
                return;
            end
            
            % Time domain plot
            if obj.plotTime
                obj.plotTimeDomain(inputSignal, desiredSignal);
            end
            
            % Frequency domain plot
            if obj.plotFreq
                obj.plotFrequencyDomain(inputSignal, desiredSignal);
            end
            
            % Spectrograms for different signals
            if obj.plotTimeFreq
                % External Noise Spectrogram
                obj.plotSingleSpectrogram(inputSignal, 'External Noise');
                
                % Noisy Speech Spectrogram
                obj.plotSingleSpectrogram(desiredSignal, 'Noisy Speech');
                
                % Clean Speech Spectrogram
                obj.plotSingleSpectrogram(obj.cleanSpeech, 'Clean Speech');
                
                % Enhanced Output Spectrogram
                obj.plotSingleSpectrogram(obj.E, 'Enhanced Speech');
            end
            
            % Filter weights evolution
            if obj.plotEvolution
                obj.plotFilterEvolution();
            end
            
            % SNR history
            if obj.plotSNRHistory
                obj.plotSNR();
            end
            
            % Reset figure window style
            set(0, 'DefaultFigureWindowStyle', 'normal');
        end
        
        function plotSingleSpectrogram(obj, signal, signalName)

            figure('Name', [signalName ' Spectrogram'], 'Color', 'white', 'Position', [100, 100, 1000, 600]);
        
            window = hamming(512); 
            noverlap = 256;
            nfft = 1024;
        
            % Improved spectrogram with better color mapping
            spectrogram(signal, window, noverlap, nfft, obj.fs, 'yaxis', 'centered');
            
            % Formatting
            title([signalName ' Time-Frequency Representation'], 'FontSize', 14, 'FontWeight', 'bold');
            xlabel('Time', 'FontWeight', 'bold');
            ylabel('Frequency (Hz)', 'FontWeight', 'bold');
            
            % Use a perceptually uniform colormap
            colormap('jet');  % More visually appealing colormap
            colorbar;
            
            % Improve readability
            set(gca, 'FontSize', 10);
        end
    end
end