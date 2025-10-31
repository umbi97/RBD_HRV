clear variables
close all
clc

% Select directory
dr_ctrl = uigetdir(pwd,'Select folder');
if isequal(dr_ctrl,0)
   error('No selected directory.')
else
   disp(['User selected ', dr_ctrl])
end

% Load your dataset
load([dr_ctrl,'\','patient_IDs.mat']);

% CAP
load([dr_ctrl,'\','table_ECG_featTOT_CAP.mat']);
table_ECG_feat(8,:) = [];
database{1} = table_ECG_feat{:,1:end-1};
class{1} = table_ECG_feat{:,end};

% TuSDi
load([dr_ctrl,'\','table_ECG_featTOT_TO.mat']);
database{2} = table_ECG_feat{:,1:end-1};
table_ECG_feat{:,end} = [zeros(15,1);1;zeros(6,1);1;0;0;ones(19,1)];
class{2} = table_ECG_feat{:,end};

% DREAMS
load([dr_ctrl,'\','table_ECG_featTOT_DREAMS.mat']);
database{3} = table_ECG_feat{:,1:end-1};
class{3} = table_ECG_feat{:,end};

% ISRUC
load([dr_ctrl,'\','table_ECG_featTOT_ISRUC.mat']);
database{4} = table_ECG_feat{:,1:end-1};
class{4} = table_ECG_feat{:,end};

aim = "RnR"; % (0=Sani+OSAS / 1=RBD)

% % Uncomment these lines if you want to do Healthy/unHealthy classification
% aim = "SnS"; % Classificazione Sani/non-Sani (0=Sani / 1=RBD+OSAS)
% class{1}(16:19) = 1;
% class{2}(4,7,9,2,18,20:22,12,15) = 1; !CONTROLLARE DATASET!
% class{3}(21:32) = 1;
% class{4}(11:69) = 1;

%% Divisione dataset HR in TR, VAL, TE
% TR{1,1} = [14,15,2:7,38:41,26:30,32:34];
% TR{2,1} = [1:3,7:9,29,30,33:35,27,38,39,42,44];
% TR{3,1} = [2:11,13];
% TR{4,1} = [5:9,1,70];
% 
% VAL{1,1} = [11:13,21:25];
% VAL{2,1} = [4,10,26,28,40,41];
% VAL{3,1} = [16:19];
% VAL{4,1} = [3,4];

TE{1,1} = [8:10,14,20,28,31,35:37];
TE{2,1} = [6,8,17,23,31,32,36,37]; %[5,6,31,32,36,37,43]; divisione vecchia con classificazione sbagliata
TE{3,1} = [1,12,14,15,11];
TE{4,1} = [2,10,71];

TR{1,1} = setdiff(1:size(database{1},1),[TE{1,1},16:19]);
TR{2,1} = setdiff(1:size(database{2},1),[TE{2,1},4,7,9,2,18,20:22,12,15]);
TR{3,1} = setdiff(1:size(database{3},1),[TE{3,1},21:32]);
TR{4,1} = setdiff(1:size(database{4},1),[TE{4,1},11:69]);

%% Divisione dataset HRO in TR, VAL, TE
% TR{1,2} = [14,15,2:7,18,19,38:41,26:30,32:34];
% TR{2,2} = [1:3,7:9,22:25,13,15:17,29,30,33:35,27,38,39,42,44];
% TR{3,2} = [2:11,13,29,31,32,22:24];
% TR{4,2} = [5:9,1,23:30,32:39,41:49,51:62,64:70,13];
% 
% VAL{1,2} = [11:13,17,21:25];
% VAL{2,2} = [4,10,19:21,26,28,40,41];
% VAL{3,2} = [16:19,27,28,30];
% VAL{4,2} = [5:9,1,23:30,32:39,41:49,51:62,64:70,13];

TE{1,2} = [TE{1,1},16];
TE{2,2} = [TE{2,1},18,22]; %12 da aggiungere se si vuole integrare PLM
TE{3,2} = [TE{3,1},21,25,26];
TE{4,2} = [TE{4,1},11,12,14,21,22,40,50,58:65];

TR{1,2} = setdiff(1:size(database{1},1),TE{1,2});
TR{2,2} = setdiff(1:size(database{2},1),[TE{2,2},2,7,12,15]);
TR{3,2} = setdiff(1:size(database{3},1),TE{3,2});
TR{4,2} = setdiff(1:size(database{4},1),TE{4,2});

%% Creo un vettore contenente i nomi delle features estratte
labels = table_ECG_feat.Properties.VariableNames;
s_1235 = [];
s_all = [];
for i = 1:size(labels,2)-1
    if ismember(i,1:2:size(table_ECG_feat,2)-1)
        s_1235 = [s_1235,convertCharsToStrings(labels{1,i})];
    else
        s_all = [s_all,convertCharsToStrings(labels{1,i})];
    end
end
labels_1235 = cellstr(s_1235);
labels_all = cellstr(s_all);

%% Ciclo for
comb = {1,2,[1,2],[1,3],[1,4],[1,2,3],[1,2,4],[1,3,4],[1,2,3,4]};
comb_name = ["CAP_","TO_","CAP+TO_","CAP+DREAMS_","CAP+ISRUC_","CAP+TO+DREAMS_","CAP+TO+ISRUC_","CAP+DREAMS+ISRUC_","CAP+TO+DREAMS+ISRUC_"];
veglia = ["1235_","all_"];
for c = 1:length(comb)
    flag = 0;
    for rip = [1,2,1,2]
        TR_temp = [];
        TE_temp = [];
        TR_IDs = [];
        TE_IDs = [];
        for d = comb{c}
            if flag < 2
                TR_temp = [TR_temp;database{d}(TR{d,1},rip:2:size(database{d},2)-1),class{d}(TR{d,1})];
                TE_temp = [TE_temp;database{d}(TE{d,1},rip:2:size(database{d},2)-1),class{d}(TE{d,1})];

                TR_IDs = [TR_IDs,IDs{d}(TR{d,1})];
                TE_IDs = [TE_IDs,IDs{d}(TE{d,1})];

                HR_HRO = "HR_";
            else
                TR_temp = [TR_temp;database{d}(TR{d,2},rip:2:size(database{d},2)-1),class{d}(TR{d,2})];
                TE_temp = [TE_temp;database{d}(TE{d,2},rip:2:size(database{d},2)-1),class{d}(TE{d,2})];

                TR_IDs = [TR_IDs,IDs{d}(TR{d,2})];
                TE_IDs = [TE_IDs,IDs{d}(TE{d,2})];

                HR_HRO = "HRO_";
            end
        end
        TR_maxmin = [max(TR_temp);min(TR_temp)];
        TR_norm = normalize(TR_temp,'range');

        TE_norm = zeros(size(TE_temp));
        for i = 1:size(TE_temp,2)
            TE_norm(:,i) = (TE_temp(:,i)-TR_maxmin(2,i))/(TR_maxmin(1,i)-TR_maxmin(2,i));
        end
        
        X = TR_norm(:,1:end-1);
        y = TR_norm(:,end);
        correlation_with_target = zeros(1, size(X,2));
        % correlation_with_target_S = zeros(1, size(X,2));
        for i = 1:size(X,2)
            correlation_with_target(i) = corr(X(:, i), y)+corr(X(:, i), y, 'Type', 'Spearman');
        end
        
        % Calculate the correlation matrix between features
        feature_correlation_matrix = corr(X);
        feature_correlation_matrix_S = corr(X, 'Type', 'Spearman');
        
        % Set a correlation threshold for feature selection
        correlation_threshold = 0.8; % Adjust as needed
        
        % Initialize a logical mask for feature selection
        feature_selection_mask = true(1, size(X, 2));
        
        % Iterate through the feature pairs and keep the one with the highest correlation with the target
        for i = 1:size(X, 2)
            for j = i+1:size(X, 2)
                if feature_selection_mask(i) && feature_selection_mask(j) && abs(feature_correlation_matrix(i, j)) > correlation_threshold || abs(feature_correlation_matrix_S(i, j)) > correlation_threshold
                    if abs(correlation_with_target(i)) >= abs(correlation_with_target(j))
                        feature_selection_mask(j) = false; % Keep feature i
                    else
                        feature_selection_mask(i) = false; % Keep feature j
                    end
                end
            end
        end
        if rip == 1
            labels_NOcorr = labels_1235(feature_selection_mask);
        else
            labels_NOcorr = labels_all(feature_selection_mask);
        end
        
        %% SAVE
        feature_selection_mask = [feature_selection_mask, true];
        TR_norm_NOcorr = TR_norm(:, feature_selection_mask);
        TE_norm_NOcorr = TE_norm(:, feature_selection_mask);
        % [TE_greater1(1,:),TE_greater1(2,:)] = find(TE_norm>1);

        % Salvataggio
        table_ECG_feat = array2table(TR_norm_NOcorr,"VariableNames",[labels_NOcorr,'CLASS'],"RowNames",TR_IDs);
        writetable(table_ECG_feat,strcat("feat_eval\",comb_name(c),"TR_",veglia(rip),HR_HRO,"NOcorr_",aim,".csv"),"WriteRowNames",true);
        table_ECG_feat = array2table(TE_norm_NOcorr,"VariableNames",[labels_NOcorr,'CLASS'],"RowNames",TE_IDs);
        writetable(table_ECG_feat,strcat("feat_eval\",comb_name(c),"TE_",veglia(rip),HR_HRO,"NOcorr_",aim,".csv"),"WriteRowNames",true);

        flag = flag+1;
    end
end