clear variables
close all
clc

% Import signals

% Import *ONE CHANNEL ONLY* from all records in folder. 

% % Select directory
% dr_ctrl = uigetdir(pwd,'Select folder');
% if isequal(dr_ctrl,0)
%    error('No selected directory.')
% else
%    disp(['User selected ', dr_ctrl])
% end
% fname = [dr_ctrl filesep rec_info(i).name];

dr = ["C:\Users\umbym\Documents\Politecnico\Magistrale\Tesi\Code\cap-sleep-database-1.0.0",...
    "C:\Users\umbym\Documents\Politecnico\Magistrale\Tesi\Code\Turin_database\CMS_DATA\n_rbd_rswa",...
    "C:\Users\umbym\Documents\Politecnico\Magistrale\Tesi\Code\DREAMS_database",...
    "C:\Users\umbym\Documents\Politecnico\Magistrale\Tesi\Code\ISRUC-SLEEP_database\Healty_RBD"];
names = ["table_ECG_featTOT_CAP.mat","table_ECG_featTOT_TO.mat","table_ECG_featTOT_DREAMS.mat","table_ECG_featTOT_ISRUC.mat";...
    "table_ECG_plot_CAP.mat","table_ECG_plot_TO.mat","table_ECG_plot_DREAMS.mat","table_ECG_plot_ISRUC.mat"];
ecg_filt_old = {};
flag = 0;
ord = 0;
for d = 2:2%length(dr)
    dr_ctrl = convertStringsToChars(dr(d));
    path = strsplit(dr_ctrl,'\');
    if strcmp(path{1,end},'cap-sleep-database-1.0.0')    
        % The following instructions select Controls records + RBD records
        rec_info = [dir([dr_ctrl filesep 'n*.edf']);dir([dr_ctrl filesep 'sd*.edf']);dir([dr_ctrl filesep 'r*.edf'])];% ;
        rec_info(17:61,:) = []; 
        note_info = [dir([dr_ctrl filesep 'n*.txt']);dir([dr_ctrl filesep 'sd*.txt']);dir([dr_ctrl filesep 'r*.txt'])];% ;
        note_info(17:61,:) = [];
    elseif strcmp(path{1,end},'n_rbd_rswa')
        % The following instructions select Controls records + RBD records
        rec_info = [dir([dr_ctrl filesep 'n*.edf']);dir([dr_ctrl filesep 'o*.edf']);dir([dr_ctrl filesep 'r*.edf'])];% ;
        note_info = [dir([dr_ctrl filesep 'n*.txt']);dir([dr_ctrl filesep 'o*.txt']);dir([dr_ctrl filesep 'r*.txt'])];% ;
    elseif strcmp(path{1,end},'DREAMS_database')
        rec_info = [dir([dr_ctrl filesep 's*.edf']);dir([dr_ctrl filesep 'e*.edf'])];% ;
        note_info = [dir([dr_ctrl filesep 'HypnogramAASM_s*.txt']);dir([dr_ctrl filesep 'Hypnogram_e*.txt'])];% ;
    elseif strcmp(path{1,end-1},'ISRUC-SLEEP_database')
        % The following instructions select Controls records + RBD records
        rec_info = [dir([dr_ctrl filesep 'H*.rec']);dir([dr_ctrl filesep 'O*.rec']);dir([dr_ctrl filesep 'R*.rec'])];% ;
        note_info = [dir([dr_ctrl filesep 'H*.txt']);dir([dr_ctrl filesep 'O*.txt']);dir([dr_ctrl filesep 'R*.txt'])];% ;
    end
    
    % Features ECG labels
    if flag == 0
        stats = ["mean","devstd","prc25","prc75","kurt","skew"];
        feat = ["HR","SDNN","RMSSD","pLF","pHF","LFHFratio","VLF","LF","HF","ApEn","SampEN","DFA1","DFA2","SD1","SD2","SD1SD2ratio","KC_05","KC_1","KC_2","LZC_05","LZC_1","LZC_2"]; %"CD"
        stad = ["1235_05","all_05","1235_1","all_1","1235_2","all_2","1235_4","all_4"];
        for i = 1:length(feat)
            for j = 1:length(stats)
                for k = 1:length(stad)
                    ECG_feat_labels{length(stats)*length(stad)*(i-1)+length(stad)*(j-1)+k} = char(stats(j)+'_'+feat(i)+'_'+stad(k));
                end
            end
        end
        ECG_feat_labels{length(ECG_feat_labels)+1} = 'CLASS';
        ECG_plot_labels{1} = 'ECG'; ECG_plot_labels{2} = 'peakR_height'; ECG_plot_labels{3} = 'peakR_locs'; ECG_plot_labels{4} = 'hyp'; ECG_plot_labels{5} = 'peakR_hyp'; ECG_plot_labels{6} = 'err';
        flag = flag+1;
    end

    ECG_feat = zeros(length(rec_info),length(ECG_feat_labels));
    ECG_plot = {};
    ECG_plot_old = {};
    ECG_plot_new = {};
    
    err_tot = 0;
    for i = 1:length(rec_info)
        disp(names(1,d))
        disp(i)
        % select file
        fname = [dr_ctrl filesep rec_info(i).name];
        aname = [dr_ctrl filesep note_info(i).name];
    
        % read records and hypnogram
        [hdr,rec] = edfreadUntilDone(fname);
        if strcmp(path{1,end},'cap-sleep-database-1.0.0')
            [hyp,~,~,~] = readscore(aname);
        elseif strcmp(path{1,end},'n_rbd_rswa')
            hyp = readscoreTurin(aname,hdr,length(rec(1))/hdr.frequency(1));
        elseif strcmp(path{1,end},'DREAMS_database')
            hyp = readscoreDREAMS(aname);
        elseif strcmp(path{1,end-1},'ISRUC-SLEEP_database')
            hyp = readscoreISRUC(aname);
        end

        idx_pos=find(strcmp(hdr.label,'Pos')==1);
        pos = rec(idx_pos,:);
        plot(pos)
    
        % ECG
        % find Channel, frequency and # of hypnogram epochs
        idx_ecg=find(strcmp(hdr.label,'ECG1ECG2')==1 | strcmp(hdr.label,'EKG')==1 | ...
            strcmp(hdr.label,'ECG')==1 | strcmp(hdr.label,'ekg')==1 | strcmp(hdr.label,'X2')==1 | strcmp(hdr.label,'25')==1);
        if isempty(idx_ecg) == 0
            if length(idx_ecg)>1
                idx_ecg=idx_ecg(1);
            end
            sig = rec(idx_ecg,:);
            sig = sig - mean(sig);
        elseif sum(strcmp(hdr.label,'ECG1')==1) + sum(strcmp(hdr.label,'ECG2')==1) == 2
            idx_ecg = find(strcmp(hdr.label,'ECG1')==1 | strcmp(hdr.label,'ECG2')==1);
            sig1 = rec(idx_ecg(1),:);
            sig1 = sig1 - mean(sig1);
            sig2 = rec(idx_ecg(2),:);
            sig2 = sig2 - mean(sig2);
            sig = sig1-sig2;
            idx_ecg = idx_ecg(1);
            clear sig1 sig2
        else
            idx_ecg = 0;
        end
    
        if idx_ecg ~= 0
            fs_ecg = hdr.frequency(idx_ecg);
            % Check sampling frequency (Hz)
            if fs_ecg<512
                % If other than 512 Hz, resample to 512 Hz
                tx=1/fs_ecg:1/fs_ecg:length(sig)/fs_ecg;
                ecg = resample(sig,tx,512);
                fs_ecg=512;
            else
                ecg=sig;
            end
            clear sig;
    
            % Calculate number of 30-s epochs
            if strcmp(path{1,end},'DREAMS_database')
                nsamp = 5*fs_ecg;
            else
                nsamp=30*fs_ecg; % qui è giusto usare fs_eeg, e non info.NumSamples(idx_eeg), perchè il segnale è stato ricampionato
            end
    
            % Replace each 4-class with 3-class and resample the hypnogram
            for j = 1:size(hyp,1)
                if hyp(j,1) == 4
                   hypa_res(1+(j-1)*nsamp : j*nsamp) = 3;
                else
                   hypa_res(1+(j-1)*nsamp : j*nsamp) = hyp(j,1);
                end
            end    
            clear hyp;
    
            % Check Hypnogram (PSG annotations) length
            if length(ecg)==length(hypa_res)
                disp('Hypnogram length ok')
            elseif length(ecg)<length(hypa_res)
                disp(['Signal length lower, subj ',rec_info(i).name(1:end-4)]);
                hypa_res(1+length(ecg):end)=[];
            else
                disp(['Hypnogram length lower, subj ',rec_info(i).name(1:end-4)]);
                ecg(1+length(hypa_res):end)=[];
            end
    
            %% Pre-process
    
            % Filter signals + Features extraction
            ecg_filt = filter_ecg_def(ecg,fs_ecg,hdr.units(idx_ecg));
            % ecg_filt_old{i} = filter_ecg_old(ecg,fs_ecg,hdr.units(idx_ecg));

            % [snr_values, timestamps] = analyze_ecg_snr(ecg, fs_ecg);
            % SNR_dB = compute_ecg_snr(ecg, fs_ecg);

            % ECG_plot = find_RR(ecg_filt,hypa_res,fs_ecg,ECG_plot,i);
            % ECG_plot_old = find_RR(ecg_filt_old,hypa_res,fs_ecg,ECG_plot_old,i);
            % ECG_plot_new = find_RR_median(ecg_filt,hypa_res,fs_ecg,ECG_plot_new,i);

            ECG_plot = find_RR_median(ecg_filt,hypa_res,fs_ecg,ECG_plot,i);
            
            % %PLOT
            % ecg = ecg_filt_old{i};%ECG_plot{paz,1};
            % locs = ECG_plot{i,3};
            % % locs_old = ECG_plot_old{paz,3};
            % % locs_new = ECG_plot_new{paz,3};
            % figure
            % plot([1:length(ECG_plot{i,1})]/fs_ecg,ECG_plot{i,1}), hold on
            % plot(locs,ECG_plot{i,1}(locs*fs_ecg), 'gpentagram')
            % title('R peaks')
            % 
            % figure
            % subplot(2,1,1)
            % plot([1:length(ecg)]/fs_ecg,ecg), hold on
            % plot(locs,ecg(locs*fs_ecg), 'gpentagram')
            % title('R peaks')
            % 
            % subplot(2,1,2)
            % plot(diff(locs)), hold on
            % title('Tacogram')

            [ECG_feat, err(1)] = featuresNsec_ECG(ECG_plot{i,3}(ECG_plot{i,5} ~= 0),fs_ecg,i,ECG_feat,30,1);
            [ECG_feat, err(2)] = featuresNsec_ECG(ECG_plot{i,3},fs_ecg,i,ECG_feat,30,2);
            [ECG_feat, err(3)] = featuresNsec_ECG(ECG_plot{i,3}(ECG_plot{i,5} ~= 0),fs_ecg,i,ECG_feat,60,3);
            [ECG_feat, err(4)] = featuresNsec_ECG(ECG_plot{i,3},fs_ecg,i,ECG_feat,60,4);
            [ECG_feat, err(5)] = featuresNsec_ECG(ECG_plot{i,3}(ECG_plot{i,5} ~= 0),fs_ecg,i,ECG_feat,120,5);
            [ECG_feat, err(6)] = featuresNsec_ECG(ECG_plot{i,3},fs_ecg,i,ECG_feat,120,6);
            [ECG_feat, err(7)] = featuresNsec_ECG(ECG_plot{i,3}(ECG_plot{i,5} ~= 0),fs_ecg,i,ECG_feat,240,7);
            [ECG_feat, err(8)] = featuresNsec_ECG(ECG_plot{i,3},fs_ecg,i,ECG_feat,240,8);
            ECG_plot{i,6} = err;
            if d == 4
                ECG_feat(i,end) = strcmp(rec_info(i).name(1),'R');
            else
                ECG_feat(i,end) = strcmp(rec_info(i).name(1),'r');
            end
        end
        clear idx_ecg ecg_filt
    end
    % % for i = 1:length(feat)
    % %     figure(i)
    % %     title(feat(i))
    % % end
    % table_ECG_feat = array2table(ECG_feat,"VariableNames",ECG_feat_labels);
    % table_ECG_plot = cell2table(ECG_plot,"VariableNames",ECG_plot_labels);
    % 
    % save("C:\Users\umbym\OneDrive\Documenti\Tesi\" + names(1,d),'table_ECG_feat')
    % save("D:\Brevetto\Work\" + names(2,d),'table_ECG_plot','-v7.3')
end