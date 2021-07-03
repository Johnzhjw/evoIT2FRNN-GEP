%%
clc;
clear;

%%
nRun = 20;
NTRACE = 25;

%%
models = {
    'FRNN'
};

modelName = {
    'FRNN'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
};

%%
algs = {
    'PCMLIA'
    'PCMLIA-Pseu'
};
for i = 1:length(algs)
    algs{i} = strrep(algs{i}, 'PCMLIA', 'PCMLIA-ADE');
end

%%
fnms = {
    'exp1_opt'
    'exp1_pseu'
};
for i = 1:length(fnms)
    fnms{i} = strrep(fnms{i}, 'exp1_opt', 'PCMLIA-ADE');
    fnms{i} = strrep(fnms{i}, 'exp1_pseu', 'PCMLIA-ADE-Pseu');
end

%%
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
    mainfils{i} = strrep(mainfils{i}, 'paraAna20200726', 'test_new');
end

%%
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

%%
probs = {
     'gnfuv-pi2'
     'gnfuv-pi3'
     'gnfuv-pi4'
     'gnfuv-pi5'
     'hungaryChickenpox'
     'SML2010-DATA-1'
     'SML2010-DATA-2'
     'traffic'
};

tmpProbName = {
     'evoBFGRNN_Predict_TimeSeries_gnfuv-pi2'
     'evoBFGRNN_Predict_TimeSeries_gnfuv-pi3'
     'evoBFGRNN_Predict_TimeSeries_gnfuv-pi4'
     'evoBFGRNN_Predict_TimeSeries_gnfuv-pi5'
     'evoBFGRNN_Predict_TimeSeries_hungaryChickenpox'
     'evoBFGRNN_Predict_TimeSeries_SML2010-DATA-1'
     'evoBFGRNN_Predict_TimeSeries_SML2010-DATA-2'
     'evoBFGRNN_Predict_TimeSeries_traffic'
};

probName = cell(length(algs), length(probs));

dims = cell(length(algs), length(probs), nRun);

for i = 1:length(probName)
    probName{i} = tmpProbName;
end

% for i = 1:length(probName)
%     for j = 1:length(probName{i})
%         probName{i}{j} = strrep(probName{i}{j}, 'Predict_', 'Predict_Stock_');
%     end
% end

nMPI = cell(1, length(algs));

nobjs = cell(1, length(algs));

%%
nPop_all = cell(1, length(algs));

maxFEs=1e5;

%%
for i = 1:1
    for j = 1:nRun
        dims{i}{1}{j} = 713;
        dims{i}{2}{j} = 713;
        dims{i}{3}{j} = 713;
        dims{i}{4}{j} = 713;
        dims{i}{5}{j} = 9495;
        dims{i}{6}{j} = 10308;
        dims{i}{7}{j} = 10308;
        dims{i}{8}{j} = 8577;
    end
end
increments = {
    0
    0
    0
    0
    25
    0
    0
    25
};
for iAlg = 2:2
    for i=1:length(probs)
        for j = 1:nRun
            dims{iAlg}{i}{j} = dims{iAlg-1}{i}{j}+increments{i};
        end
    end
end

%%
for i = 1:length(algs)
    nobjs{i} = 3;
    nPop_all{i} = 12;
    nMPI{i} = 72;
    filestrs{i} = strrep(filestrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA');
    PFstrs{i} = strrep(PFstrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA');
    PSstrs{i} = strrep(PSstrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA');
    TIMEstrs{i} = strrep(TIMEstrs{i}, 'DPCCMOLSEA', 'DPCCMOLSIA');
end

%%
tINDs    = 1:2;
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
