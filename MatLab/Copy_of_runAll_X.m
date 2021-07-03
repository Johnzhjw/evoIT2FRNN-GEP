%%

NTRACE = 25;

models = {
    'FRNN'
};

algs = {
    'test_ori31'
    'test_GFRNN'
    'test_GFRNN2'
    'test_GFRNN3'
    'test_GFRNN4'
    'test_FGRNN'
    'test_PI'
    'test_PI'
    'test_PI'
    'test_PI'
    'test_adaPI'
    'test_adaPI'
    'test_adaPI'
    'test_adaPI'
    'test_FRNN'
};

fnms = {
    'test_ori31'
    'test_GFRNN'
    'test_GFRNN2'
    'test_GFRNN3'
    'test_GFRNN4'
    'test_FGRNN'
    'test_PI'
    'test_PI'
    'test_PI'
    'test_PI'
    'test_adaPI'
    'test_adaPI'
    'test_adaPI'
    'test_adaPI'
    'test_FRNN'
};

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
    mainfils{i} = strrep(mainfils{i}, 'paraAna20200726', 'tests');
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

TIMEstrs = {
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
    'TIME_DPCCMOLSEA'
};

dims = {
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
};

probName = {
    {'evoDFRNN_Predict_0941' 'evoDFRNN_Predict_1288' 'evoDFRNN_Predict_0005'}
    {'evoGFRNN_Predict_0941' 'evoGFRNN_Predict_1288' 'evoGFRNN_Predict_0005'}
    {'evoGFRNN_Predict_0941' 'evoGFRNN_Predict_1288' 'evoGFRNN_Predict_0005'}
    {'evoGFRNN_Predict_0941' 'evoGFRNN_Predict_1288' 'evoGFRNN_Predict_0005'}
    {'evoGFRNN_Predict_0941' 'evoGFRNN_Predict_1288' 'evoGFRNN_Predict_0005'}
    {'evoFGRNN_Predict_0941' 'evoFGRNN_Predict_1288' 'evoFGRNN_Predict_0005'}
    {'evoFRNN_Predict_0941'  'evoFRNN_Predict_1288'  'evoFRNN_Predict_0005'}
    {'evoGFRNN_Predict_0941' 'evoGFRNN_Predict_1288' 'evoGFRNN_Predict_0005'}
    {'evoDFRNN_Predict_0941' 'evoDFRNN_Predict_1288' 'evoDFRNN_Predict_0005'}
    {'evoFGRNN_Predict_0941' 'evoFGRNN_Predict_1288' 'evoFGRNN_Predict_0005'}
    {'evoFRNN_Predict_0941'  'evoFRNN_Predict_1288'  'evoFRNN_Predict_0005'}
    {'evoGFRNN_Predict_0941' 'evoGFRNN_Predict_1288' 'evoGFRNN_Predict_0005'}
    {'evoDFRNN_Predict_0941' 'evoDFRNN_Predict_1288' 'evoDFRNN_Predict_0005'}
    {'evoFGRNN_Predict_0941' 'evoFGRNN_Predict_1288' 'evoFGRNN_Predict_0005'}
    {'evoFRNN_Predict_0941'  'evoFRNN_Predict_1288'  'evoFRNN_Predict_0005'}
};

probs={
    '0941.HK'
    '1288.HK'
    '0005.HK'
};

modelName = {
    'FRNN'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
};

% nObj = 2;

nRun = 20;

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

maxFEs=1e6;

%%
for i = 1:length(algs)
    for j = 1 : nRun
        dims{1}{j} = 156;
        dims{2}{j} = 165;
        dims{3}{j} = 165;
        dims{4}{j} = 195;
        dims{5}{j} = 207;
        dims{6}{j} = 275;
        dims{7}{j} = 159;
        dims{8}{j} = 210;
        dims{9}{j} = 159;
        dims{10}{j} = 278;
        dims{11}{j} = 159;
        dims{12}{j} = 210;
        dims{13}{j} = 159;
        dims{14}{j} = 278;
        dims{15}{j} = 156;
    end
    nobjs{i} = 2;
    nPop_all{i} = 100;
    if i < 11 || i == 15
        filestrs{i} = strrep(filestrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA');
        PFstrs{i} = strrep(PFstrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA');
        PSstrs{i} = strrep(PSstrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA');
        TIMEstrs{i} = strrep(TIMEstrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA');
    else
        filestrs{i} = strrep(filestrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA_MP_ADAP');
        PFstrs{i} = strrep(PFstrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA_MP_ADAP');
        PSstrs{i} = strrep(PSstrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA_MP_ADAP');
        TIMEstrs{i} = strrep(TIMEstrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA_MP_ADAP');
    end
end

tINDs    = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15];
algs     = algs(tINDs);
fnms     = fnms(tINDs);
dims     = dims(tINDs);
nobjs    = nobjs(tINDs);
nPop_all = nPop_all(tINDs);
filestrs = filestrs(tINDs);
PFstrs   = PFstrs(tINDs);
PSstrs   = PSstrs(tINDs);
TIMEstrs = TIMEstrs(tINDs);

%%
tarINDs = 1:length(algs);%15;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lenINDs = length(tarINDs);

%%
candidRand = {
     'FRNN'
};

curAlg = [];
ALL_TIMES_NN = cell(length(candidRand),1);
mean_time_NN = 0;

ALL_min_objs_run_NN = cell(length(candidRand),1);
ALL_minInd_objs_run_NN = cell(length(candidRand),1);

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
    ALL_minInd_objs_run_NN{iRnd} = all_minInd_over_run;
    plotBER_trace;
    readTIME;
    ALL_TIMES_NN{iRnd} = all_TIMEs;

    for i = 1:length(fnms)
        algs{i} = strrep(algs{i}, candidRand{iRnd}, 'PCMLEA');
        fnms{i} = strrep(fnms{i}, candidRand{iRnd}, 'PCMLEA');
    end
end

tmp_sz = 2;
table_NN_all_mean_min_ACE_run = zeros(length(candidRand)*length(probs)*tmp_sz, lenINDs);
table_NN_all_mean_minInd_ACE_run = zeros(length(candidRand)*length(probs)*tmp_sz, lenINDs);

[tmp_str_val tmp_str_i] = sort(candidRand);

for iRnd_r = 1:length(candidRand)
    iRnd = tmp_str_i(iRnd_r);
    for iExpSet = 1 : lenINDs
        for iProb = 1 : length(probs)
            for i = 1 : tmp_sz
                tmp_r = (iRnd_r-1)*length(probs)*tmp_sz+(iProb-1)*tmp_sz+i;
                table_NN_all_mean_min_ACE_run(tmp_r, iExpSet) = mean(ALL_min_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,:));
                table_NN_all_mean_minInd_ACE_run(tmp_r, iExpSet) = mean(ALL_minInd_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,:));
            end
        end
    end
end
