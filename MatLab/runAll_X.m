%%

NTRACE = 25;

models = {
    'FRNN'
};

algs = {
    'PCMLIA'
    'PCMLIA'
    'PCMLIA'
    'PCMLIA'
    'PCMLIA'
    'PCMLIA'
    'PCMLIA'
    'PCMLIA'
    'PCMLIA'
    'PCMLIA'
    'PCMLIA-Pseu'
    'PCMLIA-Pseu'
    'PCMLIA-Pseu'
    'PCMLIA-Pseu'
    'PCMLIA-Pseu'
    'PCMLIA-Pseu'
    'PCMLIA-Pseu'
    'PCMLIA-Pseu'
    'PCMLIA-Pseu'
    'PCMLIA-Pseu'
};
for i = 1:length(algs)
    algs{i} = strrep(algs{i}, 'PCMLIA', 'PCMLIA');
end

fnms = {
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
};
for i = 1:length(fnms)
    fnms{i} = strrep(fnms{i}, 'exp1', 'exp1');
end

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
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
    {156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156}%1%%%%%%%%%%%%%%%%
};

probName = {
    {'evoFRNN_Predict_0941'  'evoFRNN_Predict_1288'  'evoFRNN_Predict_0005'}
    {'evoGFRNN_Predict_0941' 'evoGFRNN_Predict_1288' 'evoGFRNN_Predict_0005'}
    {'evoDFRNN_Predict_0941' 'evoDFRNN_Predict_1288' 'evoDFRNN_Predict_0005'}
    {'evoFGRNN_Predict_0941' 'evoFGRNN_Predict_1288' 'evoFGRNN_Predict_0005'}
    {'evoGFGRNN_Predict_0941' 'evoGFGRNN_Predict_1288' 'evoGFGRNN_Predict_0005'}
    {'evoBFRNN_Predict_0941'  'evoBFRNN_Predict_1288'  'evoBFRNN_Predict_0005'}
    {'evoBGFRNN_Predict_0941' 'evoBGFRNN_Predict_1288' 'evoBGFRNN_Predict_0005'}
    {'evoBDFRNN_Predict_0941' 'evoBDFRNN_Predict_1288' 'evoBDFRNN_Predict_0005'}
    {'evoBFGRNN_Predict_0941' 'evoBFGRNN_Predict_1288' 'evoBFGRNN_Predict_0005'}
    {'evoBGFGRNN_Predict_0941' 'evoBGFGRNN_Predict_1288' 'evoBGFGRNN_Predict_0005'}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    {'evoFRNN_Predict_0941'  'evoFRNN_Predict_1288'  'evoFRNN_Predict_0005'}
    {'evoGFRNN_Predict_0941' 'evoGFRNN_Predict_1288' 'evoGFRNN_Predict_0005'}
    {'evoDFRNN_Predict_0941' 'evoDFRNN_Predict_1288' 'evoDFRNN_Predict_0005'}
    {'evoFGRNN_Predict_0941' 'evoFGRNN_Predict_1288' 'evoFGRNN_Predict_0005'}
    {'evoGFGRNN_Predict_0941' 'evoGFGRNN_Predict_1288' 'evoGFGRNN_Predict_0005'}
    {'evoBFRNN_Predict_0941'  'evoBFRNN_Predict_1288'  'evoBFRNN_Predict_0005'}
    {'evoBGFRNN_Predict_0941' 'evoBGFRNN_Predict_1288' 'evoBGFRNN_Predict_0005'}
    {'evoBDFRNN_Predict_0941' 'evoBDFRNN_Predict_1288' 'evoBDFRNN_Predict_0005'}
    {'evoBFGRNN_Predict_0941' 'evoBFGRNN_Predict_1288' 'evoBFGRNN_Predict_0005'}
    {'evoBGFGRNN_Predict_0941' 'evoBGFGRNN_Predict_1288' 'evoBGFGRNN_Predict_0005'}
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
    3
    3
    3
    3
    3
    3
    3
    3
    2
    3
    3
    3
    3
    3
    3
    3
    3
    2
    3
    3
    3
    3
    3
    3
    3
    3
    2
};

%%
nPop_all = {
    100
    120
    120
    120
    120
    120
    120
    120
    120
    100
    120
    120
    120
    120
    120
    120
    120
    120
    100
    120
    120
    120
    120
    120
    120
    120
    120
};

maxFEs=1e5;

%%
for j = 1 : nRun
    dims{1}{j} = 156;
    dims{2}{j} = 207;
    dims{3}{j} = 156;
    dims{4}{j} = 275;
    dims{5}{j} = 326;
    dims{6}{j} = 1386;
    dims{7}{j} = 1692;
    dims{8}{j} = 1386;
    dims{9}{j} = 2015;
    dims{10}{j} = 2321;
end
for iAlg = 11 : 20
    for j = 1 : nRun
        dims{iAlg}{j} = dims{iAlg-10}{j}+3;
    end
end

%%
for i = 1:length(algs)
    nobjs{i} = 2;
    nPop_all{i} = 12;
    nMPI{i} = 72;
    filestrs{i} = strrep(filestrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA');
    PFstrs{i} = strrep(PFstrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA');
    PSstrs{i} = strrep(PSstrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA');
    TIMEstrs{i} = strrep(TIMEstrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA');
end

tINDs    = 1:20;
algs     = algs(tINDs);
fnms     = fnms(tINDs);
dims     = dims(tINDs);
nobjs    = nobjs(tINDs);
nPop_all = nPop_all(tINDs);
nMPI     = nMPI(tINDs);
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
ALL_minInv_objs_run_NN = cell(length(candidRand),1);

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
    ALL_minInv_objs_run_NN{iRnd} = all_minInv_over_run;
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
table_NN_all_min_min_ACE_run = zeros(length(candidRand)*length(probs)*tmp_sz, lenINDs);
table_NN_all_min_minInv_ACE_run = cell(length(candidRand)*length(probs)*tmp_sz, lenINDs);

[tmp_str_val tmp_str_i] = sort(candidRand);

for iRnd_r = 1:length(candidRand)
    iRnd = tmp_str_i(iRnd_r);
    for iExpSet = 1 : lenINDs
        for iProb = 1 : length(probs)
            for i = 1 : tmp_sz
                tmp_r = (iRnd_r-1)*length(probs)*tmp_sz+(iProb-1)*tmp_sz+i;
                table_NN_all_mean_min_ACE_run(tmp_r, iExpSet) = mean(ALL_min_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,:));
                table_NN_all_mean_minInd_ACE_run(tmp_r, iExpSet) = mean(ALL_minInd_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,:));
                [tmp_v, tmp_i] = min(ALL_min_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,:));
                table_NN_all_min_min_ACE_run(tmp_r, iExpSet) = tmp_v;
                table_NN_all_min_minInv_ACE_run{tmp_r, iExpSet} = [tmp_i; ALL_minInv_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,tmp_i)];
            end
        end
    end
end

tmp_0941 = table_NN_all_min_min_ACE_run(1:2,:)*9.30368422;
tmp_1288 = table_NN_all_min_min_ACE_run(3:4,:)*0.45112545;
tmp_0005 = table_NN_all_min_min_ACE_run(5:6,:)*11.4176450379;

