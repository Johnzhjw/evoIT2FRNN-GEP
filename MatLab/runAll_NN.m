%%

NTRACE = 25;

allprobnames={
    'SECOM'
};

models = {
    'NN'
};

algs = {
    'PCMLEA_OFF_NO'
%     'PCMLEA_OFF_OD_001'
%     'PCMLEA_OFF_OD_01'
    'PCMLEA_OFF_ED_001'
    'PCMLEA_OFF_ED_01'
%     'PCMLEA_OFF_OI_001'
%     'PCMLEA_OFF_OI_01'
    'PCMLEA_OFF_EI_001'
    'PCMLEA_OFF_EI_01'
    'PCMLEA_ON_NO'
%     'PCMLEA_ON_OD_001'
%     'PCMLEA_ON_OD_01'
    'PCMLEA_ON_ED_001'
    'PCMLEA_ON_ED_01'
%     'PCMLEA_ON_OI_001'
%     'PCMLEA_ON_OI_01'
     'PCMLEA_ON_EI_001'
    'PCMLEA_ON_EI_01'
};
% for i = 1:length(algs)
%     algs{i} = strrep(algs{i}, 'PCMLEA', 'PCMLQAPSO');
% end

fnms = {
    'PCMLEA_OFF_NO'
%     'PCMLEA_OFF_OD_001'
%     'PCMLEA_OFF_OD_01'
    'PCMLEA_OFF_ED_001'
    'PCMLEA_OFF_ED_01'
%     'PCMLEA_OFF_OI_001'
%     'PCMLEA_OFF_OI_01'
    'PCMLEA_OFF_EI_001'
    'PCMLEA_OFF_EI_01'
    'PCMLEA_ON_NO'
%     'PCMLEA_ON_OD_001'
%     'PCMLEA_ON_OD_01'
    'PCMLEA_ON_ED_001'
    'PCMLEA_ON_ED_01'
%     'PCMLEA_ON_OI_001'
%     'PCMLEA_ON_OI_01'
     'PCMLEA_ON_EI_001'
    'PCMLEA_ON_EI_01'
};
% for i = 1:length(fnms)
%     fnms{i} = strrep(fnms{i}, 'PCMLEA', 'PCMLQAPSO');
% end

mainfils = {
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
    'paraAna20200726'
};
for i = 1:length(mainfils)
    mainfils{i} = strrep(mainfils{i}, 'paraAna20200726', 'paraAnaChaos20201011');
end

filestrs = {
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
    'DPCCMOLSEA'
};

PFfilestrs = {
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
    'PF'
};

PFstrs = {
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
    'DPCCMOLSEA_FUN'
};

PSfilestrs = {
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
    'PS'
};

PSstrs = {
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
    'DPCCMOLSEA_VAR'
};

dims = {
%     {1106 1106 1106 1106 1106 1106 1106 1106 1106 1106}%1%%%%%%%%%%%%%%%%
% %     {1107 1107 1107 1107 1107 1107 1107 1107 1107 1107}
% %     {1107 1107 1107 1107 1107 1107 1107 1107 1107 1107}
%     {1580 1580 1580 1580 1580 1574 1580 1580 1580 1580}
%     {1580 1580 1580 1580 1580 1574 1580 1580 1580 1580}
% %     {1107 1107 1107 1107 1107 1107 1107 1107 1107 1107}
% %     {1107 1107 1107 1107 1107 1107 1107 1107 1107 1107}
%     {1580 1580 1580 1580 1580 1574 1580 1580 1580 1580}
%     {1580 1580 1580 1580 1580 1574 1580 1580 1580 1580}
    {400778 400778 400778 400778 400778 397178 400778 400778 400778 400778}%10
%     {1107 1107 1107 1107 1107 1107 1107 1107 1107 1107}
%     {1107 1107 1107 1107 1107 1107 1107 1107 1107 1107}
    {401252 401252 401252 401252 401252 397646 401252 401252 401252 401252}
    {401252 401252 401252 401252 401252 397646 401252 401252 401252 401252}
%     {1107 1107 1107 1107 1107 1107 1107 1107 1107 1107}
%     {1107 1107 1107 1107 1107 1107 1107 1107 1107 1107}
    {401252 401252 401252 401252 401252 397646 401252 401252 401252 401252}
    {401252 401252 401252 401252 401252 397646 401252 401252 401252 401252}
    {401252 401252 401252 401252 401252 397646 401252 401252 401252 401252}
%     {1581 1581 1581 1581 1581 1575 1581 1581 1581 1581}
%     {1581 1581 1581 1581 1581 1575 1581 1581 1581 1581}
    {401726 401726 401726 401726 401726 398114 401726 401726 401726 401726}
    {401726 401726 401726 401726 401726 398114 401726 401726 401726 401726}
%     {1581 1581 1581 1581 1581 1575 1581 1581 1581 1581}
%     {1581 1581 1581 1581 1581 1575 1581 1581 1581 1581}
    {401726 401726 401726 401726 401726 398114 401726 401726 401726 401726}
    {401726 401726 401726 401726 401726 398114 401726 401726 401726 401726}
};

probName = {
    'Classify_NN_Indus'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
};

modelName = {
    'NN'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
};

% nObj = 2;

nRun = 10;

nMPI = {
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
    120
};

nobjs = {
    2
%     3
%     3
    3
    3
%     3
%     3
    3
    3
    2
%     3
%     3
    3
    3
%     3
%     3
    3
    3
    2
%     3
%     3
    3
    3
%     3
%     3
    3
    3
    2
};

%%
nPop_all = {
    100
%     120
%     120
    120
    120
%     120
%     120
    120
    120
    100
%     120
%     120
    120
    120
%     120
%     120
    120
    120
    100
%     120
%     120
    120
    120
%     120
%     120
    120
    120
};

%%
for i = 1:length(algs)
    nobjs{i} = 2;
    nPop_all{i} = 100;
end

%%
tarINDs = 1:10;%15;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lenINDs = length(tarINDs);

%%
candidRand = {
     'NN'
};

curAlg = [];
ALL_TIMES_NN = cell(length(candidRand),1);
mean_time_NN = 0;

ALL_min_objs_run_NN = cell(length(candidRand),1);

%%
for iRnd = 1:length(candidRand)
    for i = 1:length(fnms)
        algs{i} = strrep(algs{i}, 'PCMLEA', candidRand{iRnd});
        fnms{i} = strrep(fnms{i}, 'PCMLEA', candidRand{iRnd});
    end
    
    curAlg = candidRand{iRnd};

    readPF;
    readPF_trace;
    arrangePF;
    arrangePF_trace;
    ALL_min_objs_run_NN{iRnd} = all_min_over_run;
    plotBER_trace;
    readTIME;
    ALL_TIMES_NN{iRnd} = all_TIMEs;

    for i = 1:length(fnms)
        algs{i} = strrep(algs{i}, candidRand{iRnd}, 'PCMLEA');
        fnms{i} = strrep(fnms{i}, candidRand{iRnd}, 'PCMLEA');
    end
end

table_NN_train_mean_min_ACE_run = zeros(length(candidRand), lenINDs);
table_NN_test_mean_min_ACE_run = zeros(length(candidRand), lenINDs);
table_NN_both_mean_min_ACE_run = zeros(length(candidRand), lenINDs);

for iRnd = 1:length(candidRand)
    for iExpSet = 1 : lenINDs
        table_NN_train_mean_min_ACE_run(iRnd, iExpSet) = mean(ALL_min_objs_run_NN{iRnd}(iExpSet,1,1,1,1,:));
        table_NN_test_mean_min_ACE_run(iRnd, iExpSet) = mean(ALL_min_objs_run_NN{iRnd}(iExpSet,1,2,1,1,:));
        table_NN_both_mean_min_ACE_run(iRnd, iExpSet) = mean(ALL_min_objs_run_NN{iRnd}(iExpSet,1,3,1,1,:));
    end
end

table_NN_all_mean_min_ACE_run = zeros(lenINDs * 3, length(candidRand));

for iRow = 1 : size(table_NN_all_mean_min_ACE_run,1)
    row = floor((iRow - 1) / 3) + 1;
    arr = mod((iRow - 1), 3) + 1;
    switch arr
        case 1
            tmp = table_NN_train_mean_min_ACE_run(:,row);
        case 2
            tmp = table_NN_test_mean_min_ACE_run(:,row);
        case 3
            tmp = table_NN_both_mean_min_ACE_run(:,row);
    end
    table_NN_all_mean_min_ACE_run(iRow,:) = tmp';
end

[tmp_str_val tmp_str_i] = sort(candidRand);

table_NN_all_mean_min_ACE_run_final = table_NN_all_mean_min_ACE_run(:,tmp_str_i);
