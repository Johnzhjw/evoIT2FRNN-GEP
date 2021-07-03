%%

all_meanHVs_raw = cell(length(algs),3);

for i = 1:length(algs)
    ffile = sprintf('%s/%s/OUTPUT/',mainfils{i},fnms{i});
    fname = sprintf('MEAN_HV_TRAIN_*');
    fname = dir([ffile fname]);
    fname1= sprintf('%s\\%s',fname(1).folder,fname(1).name);
    tmp1  = importdata(fname1);
    tmp1  = tmp1.data;
    tmp   = tmp1;
    tmp(:,26) = [];
    all_meanHVs_raw{i,1} = tmp;

    ffile = sprintf('%s/%s/OUTPUT/',mainfils{i},fnms{i});
    fname = sprintf('MEAN_HV_TEST_*');
    fname = dir([ffile fname]);
    fname1= sprintf('%s\\%s',fname(1).folder,fname(1).name);
    tmp1  = importdata(fname1);
    tmp1  = tmp1.data;
    tmp   = tmp1;
    tmp(:,26) = [];
    all_meanHVs_raw{i,2} = tmp;

    ffile = sprintf('%s/%s/OUTPUT/',mainfils{i},fnms{i});
    fname = sprintf('MEAN_HV_%s*', filestrs{i});
    fname = dir([ffile fname]);
    fname1= sprintf('%s\\%s',fname(1).folder,fname(1).name);
    tmp1  = importdata(fname1);
    tmp1  = tmp1.data;
    tmp   = tmp1;
    tmp(:,26) = [];
    all_meanHVs_raw{i,3} = tmp;
end

%%

all_stdHVs_raw = cell(length(algs),3);

for i = 1:length(algs)
    ffile = sprintf('%s/%s/OUTPUT/',mainfils{i},fnms{i});
    fname = sprintf('STD_HV_TRAIN_*');
    fname = dir([ffile fname]);
    fname1= sprintf('%s\\%s',fname(1).folder,fname(1).name);
    tmp1  = importdata(fname1);
    tmp1  = tmp1.data;
    tmp   = tmp1;
    tmp(:,26) = [];
    all_stdHVs_raw{i,1} = tmp;

    ffile = sprintf('%s/%s/OUTPUT/',mainfils{i},fnms{i});
    fname = sprintf('STD_HV_TEST_*');
    fname = dir([ffile fname]);
    fname1= sprintf('%s\\%s',fname(1).folder,fname(1).name);
    tmp1  = importdata(fname1);
    tmp1  = tmp1.data;
    tmp   = tmp1;
    tmp(:,26) = [];
    all_stdHVs_raw{i,2} = tmp;

    ffile = sprintf('%s/%s/OUTPUT/',mainfils{i},fnms{i});
    fname = sprintf('STD_HV_%s*', filestrs{i});
    fname = dir([ffile fname]);
    fname1= sprintf('%s\\%s',fname(1).folder,fname(1).name);
    tmp1  = importdata(fname1);
    tmp1  = tmp1.data;
    tmp   = tmp1;
    tmp(:,26) = [];
    all_stdHVs_raw{i,3} = tmp;
end

%%

all_meanHVs = cell(length(algs),length(models),length(probs),3);

for iAlg = 1:length(algs)
    for i=1:length(models)
        for j=1:length(probs)
            for k=1:3
                all_meanHVs{iAlg,i,j,k} = ...
                all_meanHVs_raw{iAlg,k}((i-1)*length(probs)+j,:);
            end
        end
    end
end

%%

all_stdHVs = cell(length(algs),length(models),length(probs),3);

for iAlg = 1:length(algs)
    for i=1:length(models)
        for j=1:length(probs)
            for k=1:3
                all_stdHVs{iAlg,i,j,k} = ...
                all_stdHVs_raw{iAlg,k}((i-1)*length(probs)+j,:);
            end
        end
    end
end
