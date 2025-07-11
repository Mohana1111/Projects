config = struct();
config.forgettingFactor = 0.9999;
config.filterSize = 8;
config.tonalEnable = false;
config.tonalFreqs = [500, 1000.01];
config.tonalBandWidths = [20, 20];
config.plotTime = true;
config.plotFreq = true;
config.plotTimeFreq = true;
config.plotEvolution = true;
config.delta = 1e-4;
config.cleanSpeech = load("data/clean_speech.txt");
config.fs = 44100;

rls = ToolboxRLS(config);

noisy = 'data/noisy_speech.txt';  
exteral = 'data/external_noise.txt';  

d = load(noisy);      % d(n) = noisy signal = s(n) + w(n)
w = load(exteral);    % w(n) = external noise   

% d = w + config.cleanSpeech;  % Test signal, should almost fully recover clean speech -> We get SNR ~ 66.0 dB

N = length(d);

notch_x = zeros(N,1);
tic;
for i = 1:N
    if config.tonalEnable % if tonal retention is enabled, 
        x = rls.notchRecursive(w(i));  % apply notch filter
    else
        x = w(i);
    end

    notch_x(i) = x;
    rls.recursiveUpdate(x, d(i)); 
    % rls.updateSNRHistory();
end
time_done = toc;
fprintf("Processing time: %.2f seconds\n", time_done);
fprintf("Signal is processed... Evaluating Performance...\n");

% disp(size(rls.E));
% disp(size(rls.cleanSpeech));
rls.computeSNR(d);
rls.computeRNTN(d);
rls.plotAllResults(w, d);

if config.tonalEnable 
    figure;
    Nfft = length(notch_x);
    E_fft_1 = fft(notch_x);
    f = config.fs*(0:(Nfft/2))/Nfft;
    
    plot(f, abs(E_fft_1(1:Nfft/2+1)));
    title('Notched_W');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
end

